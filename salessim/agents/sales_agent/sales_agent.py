import os
import json
import tempfile
import urllib.request
from urllib.parse import unquote
from typing import List, Union
from io import BytesIO

import sys
import os
import hashlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.ai_client import AIClient
from salessim.agents.sales_agent.prompts import (
    get_system_instruction,
)
from salessim.services.http_clients import ProductLookupClient, BuyingGuideClient
from salessim.agents.utils import find_recommended_items
from common.bcolors import bcolors
from common.ai_client import VLLMResponseWrapper
import base64
from PIL import Image
# Function to encode the image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
class SalesAgent(object):

    def __init__(self, ai_client: AIClient = None, salesbot_model_params: dict = None, product_category: str = None, product_image_dir: str = None):
        self.model_params = salesbot_model_params
        self.ai_client = ai_client
        lookup_base_url = os.environ.get("LOOKUP_BASE_URL", "http://127.0.0.1:8003")
        self.buying_guide_client = BuyingGuideClient(base_url=lookup_base_url)
        self.product_catalog_client = ProductLookupClient(base_url=lookup_base_url)
        self.product_category = product_category
        self.remote_image_cache_dir = "cached_images"
        self.enable_product_images = os.environ.get("ENABLE_PRODUCT_IMAGES") == "1"
        self.product_image_dir = os.path.join(product_image_dir, product_category)
        self.intermediate_log_path = os.environ.get("INTERMEDIATE_LOG_PATH")
        self._encoded_image_cache = {}
        # Define tool schemas for OpenAI function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_buying_guide",
                    "description": "Search for buying guides and articles about product categories to help understand product features and comparisons",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for buying guide information"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "lookup_product_items",
                    "description": "Search for specific products in the store inventory based on customer requirements",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for product items"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    async def _build_recommended_products_message(self, role: str, recommended_products: list, initial_text_response: str) -> dict | None:
        content_blocks = [{"type": "text", "text": initial_text_response}]
        for entry in recommended_products:
            meta = entry.get("metadata", {}) if isinstance(entry, dict) else entry.metadata
            title = meta.get("title")
            image = meta.get("image")
            text_prefix = f"Image for product {title}:"
            if not image:
                print("No image found!!")
                continue
            resolved_url = self._resolve_image_url(image)
            if not resolved_url:
                print(f"Skipping image for {title}: could not encode image")
                continue
            content_blocks.append({"type": "text", "text": text_prefix})
            print("Adding image!!")
            content_blocks.append({"type": "image_url",
                                    "image_url": {
                                        "url": resolved_url
                                    }})
        return content_blocks

    async def cleanup(self):
        """Clean up HTTP client sessions"""
        await self.buying_guide_client.close()
        await self.product_catalog_client.close()

    def parse_chat_history(self, chat_history):
        messages = []
        for u in chat_history:
            if u["speaker"] =="Shopper":
                messages.append({"role": "user", "content": u["content"]})
            elif u["speaker"] == "Salesperson":
                if len(messages) > 0:
                    messages.append({"role": "assistant", "content": u["content"]})
        return messages

    def _build_messages(self, curr_content: Union[str, dict], chat_history: List[str]) -> List[dict]:
        """Build standardized message format for all models"""
        messages = [{"role": "system", "content": get_system_instruction(self.product_category)}]
        messages.extend(self.parse_chat_history(chat_history))
        messages.append({"role": "user", "content": curr_content})
        return messages

    def _cache_remote_image(self, url: str) -> str | None:
        """Download a remote image into a persistent cache, returning the local path."""
        try:
            suffix = os.path.splitext(url.split('?')[0])[-1] or '.jpg'
            os.makedirs(self.remote_image_cache_dir, exist_ok=True)
            cache_name = hashlib.sha256(url.encode("utf-8")).hexdigest() + suffix
            cache_path = os.path.join(self.remote_image_cache_dir, cache_name)
            if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
                return cache_path

            fd, temp_path = tempfile.mkstemp(dir=self.remote_image_cache_dir, suffix=suffix)
            os.close(fd)
            try:
                urllib.request.urlretrieve(url, temp_path)
                os.replace(temp_path, cache_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            return cache_path
        except Exception as e:
            print(f"Failed to cache remote image {url}: {e}")
            return None

    def _resolve_image_url(self, image_value: str) -> str:
        decoded_value = unquote(image_value).strip()
        if decoded_value.startswith(("http://", "https://")):
            cached_path = self._cache_remote_image(decoded_value)
            if cached_path:
                decoded_value = cached_path
            else:
                return None
        if not os.path.isabs(decoded_value):
            candidate = os.path.join(self.product_image_dir, decoded_value)
            if os.path.exists(candidate):
                decoded_value = candidate
        image_path = os.path.abspath(decoded_value)
        return self._encode_image_as_data_url(image_path)

    def _encode_image_as_data_url(self, image_path: str) -> str | None:
        cached = self._encoded_image_cache.get(image_path)
        if cached is not None:
            return cached
        try:
            with Image.open(image_path) as image:
                image.load()

                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGBA" if "A" in image.getbands() else "RGB")

                output = BytesIO()
                if image.mode == "RGBA":
                    image.save(output, format="PNG", optimize=True)
                    mime_type = "image/png"
                else:
                    image.save(output, format="JPEG", quality=85, optimize=True)
                    mime_type = "image/jpeg"

                encoded = base64.b64encode(output.getvalue()).decode("utf-8")
                data_url = f"data:{mime_type};base64,{encoded}"
                self._encoded_image_cache[image_path] = data_url
                return data_url
        except Exception as e:
            print(f"Failed to encode image {image_path}: {e}")
            return None

    def _build_product_image_message(self, product_candidates: list) -> dict | None:
        if not self.enable_product_images:
            return None

        content_blocks = []
        image_entries = []
        for candidate in product_candidates:
            image_value = candidate.metadata.get("image")
            if not image_value:
                continue
            title = candidate.metadata.get("title", "Unknown product")
            resolved_url = self._resolve_image_url(image_value)
            if not resolved_url:
                print(f"Skipping image for {title}: could not encode image")
                continue
            content_blocks.append({
                "type": "text",
                "text": f"Product image for: {title}"
            })
            content_blocks.append({
                "type": "image_url",
                "image_url": {
                    "url": resolved_url
                },
            })
            image_filename = os.path.basename(image_value)
            image_entries.append({
                "title": title,
                "image": image_filename,
                })

        if not content_blocks:
            return None

        if self.enable_product_images:
            self._log_intermediate_images(image_entries)
        return {
            "role": "user",
            "content": content_blocks
        }

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
        for key in ("repetition_penalty", "frequency_penalty", "presence_penalty", "stop", "reasoning_effort"):
            if key in self.model_params and self.model_params[key] is not None:
                extra_body[key] = self.model_params[key]
        return extra_body or None

    def _prune_previous_image_blocks(self, messages: list) -> None:
        for msg in messages:
            if hasattr(msg, "to_dict"):
                msg_dict = msg.to_dict()
                content = msg_dict.get("content")
                if isinstance(content, list):
                    msg_dict["content"] = [
                        block
                        for block in content
                        if not (isinstance(block, dict) and block.get("type") == "image_url")
                    ]
                msg_index = messages.index(msg)
                messages[msg_index] = msg_dict
                continue
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, list):
                    msg["content"] = [
                        block
                        for block in content
                        if not (isinstance(block, dict) and block.get("type") == "image_url")]


    def _log_intermediate_images(self, image_entries: list) -> None:
        if not self.intermediate_log_path:
            return
        try:
            record = {
                "event": "product_images_attached",
                "images": image_entries
            }
            with open(self.intermediate_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"{bcolors.WARNING}Failed to write intermediate log: {e}{bcolors.ENDC}")

    async def _execute_tool_call_async(self, function_name: str, function_args: dict, knowledge_used: list, all_product_candidates: list):
        """Execute tool calls asynchronously using HTTP clients"""
        print(f"{bcolors.OKGREEN}Action: {function_name}{bcolors.ENDC}")
        if isinstance(function_args, dict):
            query = function_args.get("query") or function_args.get("message") or function_args.get("product") or ""
        print(f"{bcolors.OKBLUE}Query: {query}{bcolors.ENDC}")
        if not query:
            print(f"{bcolors.WARNING}Missing query for tool call: {function_name}{bcolors.ENDC}")
            knowledge_used.append({"action": function_name, "query": "", "knowledge": ""})
            return "", None

        if function_name == "lookup_buying_guide":
            knowledge_candidates = await self.buying_guide_client.top_docs(query, k=4, product_category=self.product_category)
            knowledge = "\n---\n".join([item.page_content for item in knowledge_candidates])
            knowledge_used.append({"action": function_name, "query": query, "knowledge": knowledge})

            print(f"{bcolors.OKBLUE}Knowledge: {knowledge}{bcolors.ENDC}")
            return f"Buying guide information:\n{knowledge}", None

        elif function_name == "lookup_product_items":
            product_candidates = await self.product_catalog_client.top_docs(query, k=4, product_category=self.product_category)
            all_product_candidates.extend(product_candidates)
            knowledge = "\n---\n".join([item.page_content for item in product_candidates])
            knowledge_used.append({"action": function_name, "query": query, "knowledge": knowledge})

            print(f"{bcolors.OKBLUE}Knowledge: {knowledge}{bcolors.ENDC}")
            image_message = self._build_product_image_message(product_candidates)
            return f"Product information:\n{knowledge}", image_message

        return "", None

    async def _format_final_response(self, result: str, reasoning: str, knowledge_used: list, all_product_candidates: list) -> dict:
        """Format the final response in a standardized way"""
        recommended_items = find_recommended_items(
            all_product_candidates, result
        ) if all_product_candidates else []
        if recommended_items and self.enable_product_images:
            salesperson_response = await self._build_recommended_products_message("Salesperson", recommended_items, initial_text_response=result)
        else:
            salesperson_response = result
        all_knowledge = "\n---\n".join([k["knowledge"] for k in knowledge_used])
        
        response = {
            "speaker": "Salesperson",
            "content": salesperson_response, 
            "reasoning": reasoning,
            "knowledge": all_knowledge,
            "recommended_items": recommended_items,
            "knowledge_used": knowledge_used
        }
        return response

    async def async_generate(self, curr_content: Union[str, dict], chat_history: List[str]):
        messages = self._build_messages(curr_content, chat_history)
        knowledge_used = []
        all_product_candidates = []
        dynamic_max_tokens, dynamic_thinking_budget = self._compute_dynamic_token_limits(chat_history)
        max_token_retries = 3
        extra_body = self._build_extra_body()

        # Loop until we get a communicate action
        max_iterations = 3
        iterations = 0
        while True:
            if iterations > max_iterations:
                raise Exception("Max iterations reached, model doesn't want to communicate, just call tools.")
            iterations += 1
            messages =[m.to_dict() if isinstance(m, VLLMResponseWrapper.Message) else m for m in messages]

            token_retry = 0
            while True:
                try:
                    self._prune_previous_image_blocks(messages)
                    response = await self.ai_client.async_chat_completion(
                        messages=messages,
                        model=self.model_params['model_name'],
                        max_tokens=dynamic_thinking_budget if self.model_params['with_thinking'] else dynamic_max_tokens,
                        temperature=self.model_params['temperature'],
                        tools=self.tools,
                        tool_choice="auto",
                        extra_body=extra_body,
                    )
                    break
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

            if self.model_params['with_thinking']:
                assert 'gemma' not in self.model_params['model_name'].lower(), "Gemma does not support thinking mode."
                reasoning = response["choices"][0].message.content
                messages.append({"role": "assistant", "content": f"Reasoning: <think>\n{reasoning}\n</think>\n\n"})
                token_retry = 0
                while True:
                    try:
                        self._prune_previous_image_blocks(messages)
                        response = await self.ai_client.async_chat_completion(
                            messages=messages,
                            model=self.model_params['model_name'],
                            max_tokens=max(dynamic_max_tokens - dynamic_thinking_budget, 1),
                            temperature=0.9,
                            tools=self.tools,
                            tool_choice="auto",
                            extra_body=extra_body,
                        )
                        break
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
                response["reasoning"] = reasoning

            choice = response['choices'][0]
            reasoning = response.get('reasoning', '')
            # If there are tool calls, execute them
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                messages.append(choice.message)
                for tool_call in choice.message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    knowledge_content, image_message = await self._execute_tool_call_async(
                        function_name,
                        function_args,
                        knowledge_used,
                        all_product_candidates
                    )

                    if knowledge_content:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id or "dummy",
                            "content": knowledge_content
                        })
                    if image_message:
                        self._prune_previous_image_blocks(messages)
                        messages.append(image_message)
            else:
                # No tool calls, treat as direct communication
                return await self._format_final_response(choice.message.content, reasoning, knowledge_used, all_product_candidates)
