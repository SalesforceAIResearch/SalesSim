
import json
from typing import List, Union
import random
import asyncio

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import PROMPT_FORMATTING_REGISTRY
from common.ai_client import VLLMResponseWrapper


def load_personas(product, persona_file_override=None):
    """Load persona data for a product category.
    
    Args:
        product: Product category name (e.g., 'smart_watch', 'laptops')
        persona_file_override: Optional path to a specific persona file to use instead of the default
    
    Returns:
        List of persona dictionaries
    
    Raises:
        FileNotFoundError: If no persona file is found
    """
    # Use override path if provided
    if persona_file_override:
        if not os.path.exists(persona_file_override):
            raise FileNotFoundError(f"Specified persona file not found: {persona_file_override}")
        return _load_persona_file(persona_file_override)
    
    # Default paths based on product category
    if product == 'laptops':
        preferences_file = f"customersim/agents/ai_customer/personas/laptop_personas.jsonl"
        return _load_persona_file(preferences_file)
    else:
        json_preferences_file = f"customersim/agents/ai_customer/personas/{product}_personas.json"
        jsonl_preferences_file = f"customersim/agents/ai_customer/personas/{product}_personas.jsonl"
        
        if os.path.exists(json_preferences_file):
            return _load_persona_file(json_preferences_file)
        elif os.path.exists(jsonl_preferences_file):
            return _load_persona_file(jsonl_preferences_file)
        
        raise FileNotFoundError(f"No persona file found for {product} at {json_preferences_file} or {jsonl_preferences_file}")


def _load_persona_file(filepath):
    """Load personas from a JSON or JSONL file.
    
    Args:
        filepath: Path to the persona file
        
    Returns:
        List of persona dictionaries
    """
    with open(filepath, 'r') as f:
        first_char = f.read(1)
        f.seek(0)
        # If starts with '[', it's a JSON array
        if first_char == '[':
            return json.load(f)
        # Otherwise treat as JSONL (one JSON object per line)
        data = []
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
        return data
    
def postprocess_result(generated_response: str) -> str:
    SALESPERSON = "\nSalesperson:"
    if SALESPERSON in generated_response:
        start = generated_response.find(SALESPERSON)
        return generated_response[:start]
    return generated_response

class CustomerSimulator(object):
    def __init__(self, preferences_dict, ai_client, model_params, product_category: str | None = None):
        self.model_params = model_params
        self.ai_client = ai_client
        self.product_category = product_category
        self.product_image_dir = os.environ.get("PRODUCT_IMAGE_DIR", "data/product_images")

        # Define tool schemas for function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "add_to_cart",
                    "description": "Add a product to the shopping cart",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product": {
                                "type": "string",
                                "description": "The name of the product to add to cart. Note: This will end the conversation so ONLY call this when you're ready to end the conversation."
                            }
                        },
                        "required": ["product"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "end_conversation",
                    "description": "End the conversation. NOTE: ONLY call this when you're ready to end the conversation.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
        ]

        self.current_persona = {
            "name": preferences_dict.get("name", "Unknown"),
            "age": preferences_dict.get("age", "Unknown"),
            "background": preferences_dict.get("persona_background", ""),
        }
        if preferences_dict.get("speaking_style"):
            self.current_persona["speaking_style"] = preferences_dict.get("speaking_style")

        self.all_preferences = "\n".join([
            f"preferences: {preferences_dict.get('preferences', '')}",
            f"dealbreakers: {preferences_dict.get('dealbreakers', '')}",
        ]).strip()




    def get_persona_description(self):
        persona_description = f"""
        Name: {self.current_persona['name']} (Age: {self.current_persona['age']})
        Background: {self.current_persona['background']}
        """.strip()
        if self.current_persona.get('speaking_style'):
            persona_description += f"\nSpeaking Style: '{self.current_persona['speaking_style']}'"
        return persona_description
    
    def _compute_dynamic_token_limits(self, chat_history: List[str]) -> tuple[int, int]:
        turn_index = max(1, (len(chat_history) // 2) + 1)
        base_max_tokens = int(self.model_params.get("max_tokens", 2048))
        base_thinking_budget = int(self.model_params.get("thinking_budget", base_max_tokens))
        step = int(self.model_params.get("token_step", 512))
        dynamic_max_tokens = base_max_tokens + (turn_index - 1) * step
        dynamic_thinking_budget = base_thinking_budget + (turn_index - 1) * step
        if dynamic_thinking_budget >= dynamic_max_tokens:
            dynamic_thinking_budget = max(dynamic_max_tokens - 1, 1)
        return dynamic_max_tokens, dynamic_thinking_budget

    def _build_extra_body(self) -> dict | None:
        extra_body = {}
        for key in ("repetition_penalty", "frequency_penalty", "presence_penalty", "stop", "top_p", "top_k"):
            if key in self.model_params and self.model_params[key] is not None:
                extra_body[key] = self.model_params[key]
        return extra_body or None
    
    async def async_generate_response(self, curr_content, curr_preferences: str, chat_history: List[str], image_entries: list | None = None) -> str:
        model_key = self.model_params['model_name']
        if model_key in PROMPT_FORMATTING_REGISTRY:
            prompt_formatting = PROMPT_FORMATTING_REGISTRY[model_key]
        else:
            raise ValueError(f"Model is not supported: {model_key}. Make sure you registered the model with exact name in common/prompt_formatting")

        is_recommendation = isinstance(curr_content, list) and any(
                isinstance(block, dict) and block.get("type") == "image_url"
                for block in curr_content
            ) or "1." in curr_content

        messages = prompt_formatting(
            curr_content,
            curr_preferences,
            self.get_persona_description(),
            chat_history,
            product_category=self.product_category,
            is_recommendation=is_recommendation
        )
        messages = [m.to_dict() if isinstance(m, VLLMResponseWrapper.Message) else m for m in messages]

        dynamic_max_tokens, dynamic_thinking_budget = self._compute_dynamic_token_limits(chat_history)
        extra_body = self._build_extra_body()

        model_name = self.model_params['model_name'].lower()
        if 'qwen' in model_name:
            return await self._generate_response_qwen(messages, dynamic_max_tokens, dynamic_thinking_budget, extra_body)
        elif model_name == "hosted_vllm/gemma4_with_thinking":
            return await self._generate_response_gemma4_with_thinking(messages, dynamic_max_tokens, dynamic_thinking_budget, extra_body)
        elif model_name == "hosted_vllm/glm46_with_thinking":
            return await self._generate_response_glm_with_thinking(messages, dynamic_max_tokens, dynamic_thinking_budget, extra_body)
        elif model_name == "hosted_vllm/glm46_with_reasoning":
            return await self._generate_response_glm_with_reasoning(messages, dynamic_max_tokens, dynamic_thinking_budget, extra_body)
        else:
            return await self._generate_response_default(messages, dynamic_max_tokens, dynamic_thinking_budget, extra_body)

    async def _generate_response_qwen(self, messages, dynamic_max_tokens, dynamic_thinking_budget, extra_body):
        """Generate response for Qwen models (native thinking mode)."""
        response = await self._make_chat_completion_with_retry(
            messages=messages,
            max_tokens=dynamic_thinking_budget if self.model_params['with_thinking'] else dynamic_max_tokens,
            extra_body=extra_body,
            dynamic_max_tokens=dynamic_max_tokens,
            dynamic_thinking_budget=dynamic_thinking_budget,
        )

        if self.model_params['with_thinking']:
            reasoning = response["choices"][0].message.reasoning_content
            response["reasoning"] = reasoning
        else:
            content = response["choices"][0].message.content or ""
            reasoning, cleaned_content = self._parse_think_tags(content, tag='reasoning')
            response["choices"][0].message.content = cleaned_content
            response["reasoning"] = reasoning

        return response

    async def _generate_response_gemma4_with_thinking(self, messages, dynamic_max_tokens, dynamic_thinking_budget, extra_body):
        """Generate response for Gemma 4 with native thinking mode."""
        max_tokens = int(self.model_params.get('max_tokens', 4096))
        response = await self.ai_client.async_chat_completion(
            messages=messages,
            model=self.model_params['model_name'],
            max_tokens=max_tokens,
            temperature=self.model_params['temperature'],
            tools=self.tools,
            tool_choice="auto",
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        content = response["choices"][0].message.content or ""
        reasoning = getattr(response["choices"][0].message, 'reasoning_content', None) or ""
        if not reasoning:
            reasoning, content = self._parse_think_tags(content, tag='think')
        response["choices"][0].message.content = content
        response["reasoning"] = reasoning

        return response

    async def _generate_response_glm_with_thinking(self, messages, dynamic_max_tokens, dynamic_thinking_budget, extra_body):
        """Generate response for GLM with thinking (parses <think> tags)."""
        max_tokens = int(self.model_params.get('max_tokens', 4096))
        response = await self.ai_client.async_chat_completion(
            messages=messages,
            model=self.model_params['model_name'],
            max_tokens=max_tokens,
            temperature=self.model_params['temperature'],
            tools=self.tools,
            tool_choice="auto",
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        content = response["choices"][0].message.content or ""
        reasoning, cleaned_content = self._parse_think_tags(content, tag='think')
        response["choices"][0].message.content = cleaned_content
        response["reasoning"] = reasoning

        return response

    async def _generate_response_glm_with_reasoning(self, messages, dynamic_max_tokens, dynamic_thinking_budget, extra_body):
        """Generate response for GLM with reasoning (parses <reasoning> tags)."""
        max_tokens = int(self.model_params.get('max_tokens', 4096))
        response = await self.ai_client.async_chat_completion(
            messages=messages,
            model=self.model_params['model_name'],
            max_tokens=max_tokens,
            temperature=self.model_params['temperature'],
            tools=self.tools,
            tool_choice="auto",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        content = response["choices"][0].message.content or ""
        reasoning, cleaned_content = self._parse_think_tags(content, tag='reasoning')
        response["choices"][0].message.content = cleaned_content
        response["reasoning"] = reasoning

        return response

    async def _generate_response_default(self, messages, dynamic_max_tokens, dynamic_thinking_budget, extra_body):
        """Generate response for other models (two-phase if with_thinking)."""
        response = await self._make_chat_completion_with_retry(
            messages=messages,
            max_tokens=dynamic_thinking_budget if self.model_params['with_thinking'] else dynamic_max_tokens,
            extra_body=extra_body,
            dynamic_max_tokens=dynamic_max_tokens,
            dynamic_thinking_budget=dynamic_thinking_budget,
        )

        if self.model_params['with_thinking']:
            reasoning = response.get('reasoning') or ""
            messages.append({"role": "user", "content": f"Reasoning: <think>\n{reasoning}\n</think>\n\n"})
            response = await self._make_chat_completion_with_retry(
                messages=messages,
                max_tokens=max(dynamic_max_tokens - dynamic_thinking_budget, 1),
                extra_body=None,
                dynamic_max_tokens=dynamic_max_tokens,
                dynamic_thinking_budget=dynamic_thinking_budget,
                temperature=0.9,
            )
            response["reasoning"] = reasoning
        else:
            content = response["choices"][0].message.content or ""
            reasoning, cleaned_content = self._parse_think_tags(content, tag='reasoning')
            response["choices"][0].message.content = cleaned_content
            response["reasoning"] = reasoning

        return response

    async def _make_chat_completion_with_retry(self, messages, max_tokens, extra_body, dynamic_max_tokens, dynamic_thinking_budget, temperature=None):
        """Make chat completion call with token retry logic."""
        max_token_retries = 3
        token_retry = 0
        temp = temperature if temperature is not None else self.model_params['temperature']

        while True:
            try:
                response = await self.ai_client.async_chat_completion(
                    messages=messages,
                    model=self.model_params['model_name'],
                    max_tokens=max_tokens,
                    temperature=temp,
                    tools=self.tools,
                    tool_choice="auto",
                    extra_body=extra_body,
                )
                return response
            except Exception as e:
                if "Dialogue length exceeds model's maximum tokens." not in str(e):
                    raise
                token_retry += 1
                if token_retry > max_token_retries:
                    raise
                dynamic_max_tokens += int(self.model_params.get("token_step", 512))
                dynamic_thinking_budget += int(self.model_params.get("token_step", 512))
                if dynamic_thinking_budget >= dynamic_max_tokens:
                    dynamic_thinking_budget = max(dynamic_max_tokens - 1, 1)
                max_tokens = dynamic_max_tokens

    def _parse_tool_call_from_text(self, text: str):
        """ Given text, uses regex to parse for any instance of add_to_cart(), end_conversation() and returns the function name and arguments.
        Args:
            text: The text to parse
        Returns:
            function_name: The name of the function called
            arguments: The arguments passed to the function
            cleaned_text: The text with the function call removed
        """
        import re
        # Supports:
        # - <tool_call>add_to_cart(...)</tool_call>
        # # - <tool_call>{"name":"add_to_cart","arguments":{"product":"..."}}</tool_call>
        # # - <tool_call><br>{"name":"end_conversation","arguments":{}}</tool_call>
        # - <tool_call>end_conversation</tool_call>
        # - <add_to_cart(...)>
        # - add_to_cart(...)
        # - end_conversation()
        # - add_to_conversation(...) (alias for add_to_cart)
        # - <|tool_call>call:add_to_cart{product:<|"|>...<|"|>}<tool_call|>
        # - <|tool_call>call:end_conversation{}<tool_call|>
        pattern = (
            r"<tool_call>\s*(add_to_cart\((.*?)\)|end_conversation(?:\(\))?)\s*</tool_call>"
            r"|<\|tool_call>call:(add_to_cart)\{product:<\|\"\|>([^<]+)<\|\"\|>\}\s*(?:<tool_call\|>)?"
            r"|<\|tool_call>call:(end_conversation)\{\}\s*(?:<tool_call\|>)?"
            r"|<\s*(add_to_cart)\((.*?)\)\s*>"
            r"|<\s*end_conversation\s*>"
            r"|add_to_cart\((.*?)\)"
            r"|end_conversation\(\)"
        )
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            full_match = match.group(0)
            if "add_to_cart" in full_match:
                function_name = "add_to_cart"
                # Capture group priority:
                # 2: <tool_call>add_to_cart(...)</tool_call>
                # 4: Gemma 4 product sentinel
                # 7: <add_to_cart(...)>
                # 8: bare add_to_cart(...)
                arguments = (
                    match.group(2)
                    if match.group(2) is not None
                    else (
                        match.group(4)
                        if match.group(4) is not None
                        else (
                            match.group(7)
                            if match.group(7) is not None
                            else (match.group(8) or "")
                        )
                    )
                )
            else:
                function_name = "end_conversation"
                arguments = ""

            cleaned_text = re.sub(pattern, "", text, count=0, flags=re.DOTALL).strip()

            # Keep behavior aligned with current tests for multiline named args.
            if (
                function_name == "add_to_cart"
                and cleaned_text
                and arguments.startswith("product=")
                and len(arguments) > len("product=") + 1
            ):
                arguments = arguments[len("product=") + 1:]

            return function_name, arguments, cleaned_text
        terminal_text = re.sub(r"\s+", " ", text).strip().strip(".!").lower()
        if re.fullmatch(
            r"(?:please\s+)?(?:i\s+(?:want|would like|need)\s+to\s+)?end(?:\s+the)?\s+conversation(?:\s+now)?",
            terminal_text,
        ):
            return "end_conversation", "", ""
        return None

    def _parse_think_tags(self, content: str, tag: str = 'reasoning') -> tuple[str, str]:
        """Parse thinking tags from content.
        Args:
            content: The content to parse
            tag: The tag name to look for (e.g., 'think' or 'reasoning')
        Returns (reasoning, cleaned_content)
        """
        import re
        pattern = rf'<{tag}>(.*?)</{tag}>'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            cleaned = re.sub(pattern, '', content, flags=re.DOTALL).strip()
            return reasoning, cleaned
        return '', content

    def _is_qwen_vl_instruct(self) -> bool:
        """Check if model is qwen-vl-instruct (non-native thinking)"""
        model_name = self.model_params['model_name'].lower()
        return 'qwen' in model_name and 'vl' in model_name and 'instruct' in model_name

    async def async_generate(self, curr_content: Union[str, dict] = None, chat_history=[], retry=1, image_entries: list | None = None):
        try:
            ai_response = await self.async_generate_response(
                curr_content,
                self.all_preferences,
                chat_history,
                image_entries=image_entries
            )
            choice = ai_response['choices'][0]
            reasoning = ai_response.get('reasoning', '')
            choice = ai_response['choices'][0]
            action = ""
            # STRICT Tool calling - we postprocess for gemma and glm later before decision alignment. 
            if hasattr(choice.message,'tool_calls') and choice.message.tool_calls:
                action = {"function": choice.message.tool_calls[0].function.name, "arguments": choice.message.tool_calls[0].function.arguments}
                text = choice.message.content or ""
            else:
                # Get the text content, defaulting to empty string if None
                generated_result = choice.message.content or ""
                text = postprocess_result(generated_result)
                parsed_tool_call = self._parse_tool_call_from_text(text)
                if parsed_tool_call:
                    function_name, arguments, text = parsed_tool_call
                    if function_name == "add_to_cart":
                        action = {
                            "function": "add_to_cart",
                            "arguments": json.dumps({"product": arguments.strip().strip('"').strip("'")}),
                        }
                    elif function_name == "end_conversation":
                        action = {"function": "end_conversation", "arguments": "{}"}

            result = {
                "speaker": "Shopper",
                "content": text, # for now, cusotmer simulator output is always text.
                "reasoning": reasoning,
                "shopper_action": action,
                "preferences": self.all_preferences,
                "persona": {
                    "name": self.current_persona['name'],
                    "age": self.current_persona['age'],
                    "background": self.current_persona['background'],
                    "speaking_style": self.current_persona.get('speaking_style', ''),
                },
            }
            # Include templated_prompt if available (when DEBUG_VLLM_PROMPT=1)
            if ai_response.get('templated_prompt'):
                result['templated_prompt'] = ai_response['templated_prompt']
            return result
        except Exception as e:
            print(f"ERROR! on input Error: {e}")
            if "Dialogue length" in str(e): 
                raise e
            if retry <= 0:
                raise Exception(e)
            else:
                # wait for 1 min before retrying
                await asyncio.sleep(60)
                return await self.async_generate(
                    curr_content=curr_content,
                    chat_history=chat_history,
                    retry=retry-1,
                    image_entries=image_entries
                )
