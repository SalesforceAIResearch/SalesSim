from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import os
import asyncio
import aiohttp
from aiohttp import client_exceptions
import httpx
import json
import time
import logging
from litellm import acompletion
from transformers import AutoTokenizer
# Semaphore to limit concurrent API calls (used in batch simulations)
MAX_CONCURRENT_API_CALLS = 1000
api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
logger = logging.getLogger(__name__)


def _summarize_messages(messages: List[Dict[str, str]]) -> dict:
    summary = {
        "message_count": len(messages),
        "roles": {},
        "text_chars_total": 0,
        "image_count": 0,
        "image_url_chars_total": 0,
        "image_url_chars_max": 0,
    }
    for msg in messages:
        role = msg.get("role", "unknown")
        summary["roles"][role] = summary["roles"].get(role, 0) + 1
        content = msg.get("content")
        if isinstance(content, str):
            summary["text_chars_total"] += len(content)
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text", "")
                summary["text_chars_total"] += len(text)
            elif block_type == "image_url":
                image_url = block.get("image_url", {})
                if isinstance(image_url, dict):
                    url = image_url.get("url", "")
                else:
                    url = image_url or ""
                url_len = len(url)
                summary["image_count"] += 1
                summary["image_url_chars_total"] += url_len
                summary["image_url_chars_max"] = max(summary["image_url_chars_max"], url_len)
    return summary


def _redact_messages(messages: List[Dict[str, str]], max_text_chars: int = 500) -> List[Dict[str, str]]:
    redacted = []
    for msg in messages:
        entry = dict(msg)
        content = entry.get("content")
        if isinstance(content, str):
            if len(content) > max_text_chars:
                entry["content"] = content[:max_text_chars] + "...<truncated>"
            redacted.append(entry)
            continue
        if not isinstance(content, list):
            redacted.append(entry)
            continue
        new_blocks = []
        for block in content:
            if not isinstance(block, dict):
                new_blocks.append(block)
                continue
            if block.get("type") == "image_url":
                image_url = block.get("image_url", {})
                if isinstance(image_url, dict):
                    redacted_block = dict(block)
                    redacted_block["image_url"] = dict(image_url)
                    redacted_block["image_url"]["url"] = "<redacted>"
                    new_blocks.append(redacted_block)
                else:
                    new_blocks.append({"type": "image_url", "image_url": {"url": "<redacted>"}})
            elif block.get("type") == "text":
                text = block.get("text", "")
                if len(text) > max_text_chars:
                    text = text[:max_text_chars] + "...<truncated>"
                new_blocks.append({"type": "text", "text": text})
            else:
                new_blocks.append(block)
        entry["content"] = new_blocks
        redacted.append(entry)
    return redacted


def _log_vllm_500(messages: List[Dict[str, str]], model: str, error: Exception) -> None:
    if os.environ.get("LOG_VLLM_500_SUMMARY") != "1" and os.environ.get("LOG_VLLM_500_PAYLOAD") != "1":
        return
    record = {
        "ts": time.time(),
        "model": model,
        "error": str(error),
    }
    if os.environ.get("LOG_VLLM_500_SUMMARY") == "1":
        record["summary"] = _summarize_messages(messages)
    if os.environ.get("LOG_VLLM_500_PAYLOAD") == "1":
        record["messages"] = _redact_messages(messages)
    log_path = os.environ.get("VLLM_500_DEBUG_LOG", "runs/vllm_500_debug.jsonl")
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as log_error:
        logger.warning("Failed to write VLLM 500 debug log: %s", log_error)


def _log_responses_api_error(
    messages: List[Dict[str, str]],
    model: str,
    status: int,
    url: str,
    error_text: str,
    request_id: str | None = None,
) -> None:
    record = {
        "ts": time.time(),
        "model": model,
        "status": status,
        "url": url,
        "request_id": request_id,
        "error": error_text,
        "summary": _summarize_messages(messages),
    }
    if os.environ.get("LOG_RESPONSES_API_PAYLOAD") == "1":
        record["messages"] = _redact_messages(messages)
    log_path = os.environ.get("RESPONSES_API_ERROR_LOG", "runs/responses_api_errors.jsonl")
    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as log_error:
        logger.warning("Failed to write Responses API error log: %s", log_error)


class VLLMResponseWrapper:
    """Wrapper classes to make VLLM dictionary responses compatible with object access patterns"""

    class ToolCall:
        def __init__(self, tool_call_dict):
            self.id = tool_call_dict.get('id')
            self.type = tool_call_dict.get('type')
            self.function = self.Function(tool_call_dict.get('function', {}))
        def to_dict(self):
            return {
                "id": self.id,
                "type": self.type,
                "function": self.function.to_dict(),
            }
        class Function:
            def __init__(self, function_dict):
                self.name = function_dict.get('name')
                self.arguments = function_dict.get('arguments')
            def to_dict(self):
                return {"name": self.name, "arguments": self.arguments}

    class Message:
        def __init__(self, message_dict):
            self.content = message_dict.get('content')
            self.role = message_dict.get('role')
            self.reasoning_content = message_dict.get('reasoning_content')
            self.thinking = message_dict.get('thinking')

            # Handle tool_calls
            tool_calls_data = message_dict.get('tool_calls', [])
            if tool_calls_data:
                self.tool_calls = [VLLMResponseWrapper.ToolCall(tc) for tc in tool_calls_data]
            else:
                self.tool_calls = None
        def to_dict(self):
            d = {
                "role": self.role,
                "content": self.content,
            }
            if self.reasoning_content is not None:
                d["reasoning_content"] = self.reasoning_content
            if self.thinking is not None:
                d["thinking"] = self.thinking
            if self.tool_calls is not None:
                d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
            return d
    class Choice:
        def __init__(self, choice_dict):
            self.index = choice_dict.get('index', 0)
            self.finish_reason = choice_dict.get('finish_reason')
            self.text = choice_dict.get('text')
            self.message = VLLMResponseWrapper.Message(choice_dict.get('message', {}))
        
        def to_dict(self):
            d = {
                "index": self.index,
                "finish_reason": self.finish_reason,
                "message": self.message.to_dict(),
            }
            if self.text is not None:
                d["text"] = self.text
            return d

    @staticmethod
    def wrap_choices(choices_list):
        """Convert a list of choice dictionaries to wrapped Choice objects"""
        return [VLLMResponseWrapper.Choice(choice) for choice in choices_list]


class AIClient(ABC):
    """Abstract base class for AI clients"""

    def __init__(self):
        self.total_cost = 0.0


    @abstractmethod
    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None,
        extra_body: Optional[dict] = None,
    ) -> dict:
        """Generate a chat completion response asynchronously

        Returns:
            dict: {
                'choices': str,    # The actual response content
                'reasoning': str,  # The reasoning/thinking process (empty string if not available)
            }
        """
        pass


class VLLMClient(AIClient):
    """VLLM client implementation for local and remote VLLM servers, including Qwen3 support"""

    # Class-level tokenizer cache to avoid reloading
    _tokenizer_cache: Dict[str, AutoTokenizer] = {}

    # Whitelist of models that are trusted to use trust_remote_code=True
    # Only these models are allowed to execute remote code during tokenizer loading
    _TRUSTED_MODELS_FOR_REMOTE_CODE = {
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
        "google/gemma-3-4b-it",
        "zai-org/GLM-4.6V-Flash",
    }

    def __init__(self, api_key: str = None, organization: str = None, base_url: str = None,
                 custom_api_key: str = None, custom_api_key_env: str = None, extra_headers: dict = None, **kwargs):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.chat_completions_url = f"{base_url}/chat/completions"
        self._session = None

    def _get_tokenizer(self, model: str) -> Optional[AutoTokenizer]:
        """Get or load a tokenizer for the given model (cached)."""
        if model not in self._tokenizer_cache:
            try:
                if "qwen" in model:
                    model = "Qwen/Qwen3-VL-8B-Instruct"

                # Only allow trust_remote_code for whitelisted models
                use_trust_remote_code = model in self._TRUSTED_MODELS_FOR_REMOTE_CODE

                if not use_trust_remote_code:
                    logger.info(f"Loading tokenizer for {model} without trust_remote_code (not in whitelist)")

                self._tokenizer_cache[model] = AutoTokenizer.from_pretrained(
                    model,
                    trust_remote_code=use_trust_remote_code
                )
            except Exception as e:
                logger.warning(f"Failed to load tokenizer for {model}: {e}")
                return None
        return self._tokenizer_cache[model]

    async def _get_session(self):
        """Get or create an aiohttp ClientSession for connection reuse"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _should_retry_responses_error(self, status: int) -> bool:
        return status in {400, 408, 409, 429} or status >= 500

    def _responses_retry_delay(self, attempt: int, retry_sleep_s: float) -> float:
        return retry_sleep_s * (2 ** attempt)

    async def async_chat_completion(self, messages: list, model: str, max_tokens: int, temperature: float, tools: Optional[List[dict]] = None, tool_choice: Optional[str] = None, extra_body: Optional[dict] = None) -> dict:
        """Generate a chat completion response using VLLM asynchronously, with concurrency control via semaphore."""
        async with api_semaphore:
            return await self._async_chat_completion(messages, model, max_tokens, temperature, tools, tool_choice, extra_body)


    async def _async_chat_completion(self, messages: List[Dict[str, str]], model: str, max_tokens: int, temperature: float, tools: Optional[List[dict]] = None, tool_choice: Optional[str] = None, extra_body: Optional[dict] = None) -> dict:
        """Generate a chat completion response using VLLM asynchronously"""

        # Strip provider prefix (e.g., "hosted_vllm/") from model name for vLLM API
        if "/" in model:
            model = model.split("/", 1)[-1]

        # Normalize null content in tool calls. to empty string to avoid template/API issues
        normalized_messages = []
        for msg in messages:
            m = msg.to_dict() if hasattr(msg, 'to_dict') else dict(msg)
            if m.get('content') is None:
                m['content'] = ""
            normalized_messages.append(m)
        messages = normalized_messages
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature, 
            "tools": tools,
            "tool_choice": tool_choice,
        }
        if extra_body:
            data.update(extra_body)

        max_retries = int(os.environ.get("VLLM_MAX_RETRIES", "5"))
        retry_sleep_s = float(os.environ.get("VLLM_RETRY_SLEEP_S", "2"))
        for attempt in range(max_retries + 1):
            try:
                session = await self._get_session()
                async with session.post(self.chat_completions_url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    result = await response.json()

                    # Debug: print raw response if DEBUG_VLLM_RESPONSE is set
                    if os.environ.get("DEBUG_VLLM_RESPONSE") == "1":
                        print("=" * 80)
                        print("=== RAW VLLM RESPONSE ===")
                        print("=" * 80)
                        print(json.dumps(result, indent=2, default=str))
                        print("=" * 80)

                    message = result["choices"][0]["message"]
                    content = message["content"]
                    reasoning = ""

                    # Extract reasoning if available (for models like Qwen3 with enable_thinking)
                    if "reasoning_content" in message:
                        reasoning = message["reasoning_content"]
                    elif "thinking" in message:
                        reasoning = message["thinking"]
                    return {
                        'choices': VLLMResponseWrapper.wrap_choices(result["choices"]),
                        'reasoning': reasoning,
                    }
            except aiohttp.ClientResponseError as e:
                # Retry on 5xx server errors
                if 500 <= e.status < 600 and attempt < max_retries:
                    _log_vllm_500(messages, model, e)
                    await asyncio.sleep(retry_sleep_s * (attempt + 1))
                    continue
                if 500 <= e.status < 600:
                    _log_vllm_500(messages, model, e)
                # Retry on 400 Bad Request (can be transient with VLLM)
                if e.status == 400 and attempt < max_retries:
                    logger.warning(f"VLLM 400 Bad Request (attempt {attempt + 1}/{max_retries + 1}), retrying in {retry_sleep_s * (attempt + 1)}s...")
                    _log_vllm_500(messages, model, e)  # Reuse logging for debug
                    await asyncio.sleep(retry_sleep_s * (attempt + 1))
                    continue
                logger.error(f"VLLM API error (status {e.status}): {str(e)}")
                raise Exception(f"VLLM API error: {str(e)}")
            except (client_exceptions.ServerDisconnectedError, client_exceptions.ClientConnectionError) as e:
                if attempt < max_retries:
                    _log_vllm_500(messages, model, e)
                    await asyncio.sleep(retry_sleep_s * (attempt + 1))
                    continue
                _log_vllm_500(messages, model, e)
                logger.error(f"VLLM connection error: {str(e)}")
                raise Exception(f"VLLM API error: {str(e)}")
            except Exception as e:
                logger.error(f"VLLM API error: {str(e)}")
                raise Exception(f"VLLM API error: {str(e)}")

    async def close(self):
        """Close the aiohttp session to clean up resources"""
        if self._session and not self._session.closed:
            await self._session.close()


class LiteLLMClient(AIClient):
    """LiteLLM client implementation supporting multiple providers"""

    def __init__(self, api_key: str = None, organization: str = None, base_url: str = None,
                 custom_api_key: str = None, custom_api_key_env: str = None, extra_headers: dict = None, **kwargs):
        super().__init__()

        # Store configuration for LiteLLM
        self.config = {}
        # API key handling
        if api_key:
            self.config['api_key'] = api_key
        elif os.environ.get("OPENAI_API_KEY"):
            self.config['api_key'] = os.environ.get("OPENAI_API_KEY")

        # Organization
        if organization:
            self.config['organization'] = organization

        # Base URL
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if resolved_base_url:
            self.config['base_url'] = resolved_base_url
        self.base_url = resolved_base_url

        # Custom headers handling
        headers = {}
        if extra_headers:
            headers.update(extra_headers)

        # Custom API key for special authentication
        self.custom_api_key = None
        if custom_api_key:
            headers["X-Api-Key"] = custom_api_key
            self.custom_api_key = custom_api_key
        elif custom_api_key_env:
            custom_key_env_name = custom_api_key_env
            custom_key = os.environ.get(custom_key_env_name)
            if custom_key:
                headers["X-Api-Key"] = custom_key
                self.custom_api_key = custom_key

        if headers:
            self.config['extra_headers'] = headers

        self._session = None

        # Store additional kwargs for special model configurations
        for key, value in kwargs.items():
            self.config[key] = value

    async def _get_session(self):
        """Get or create an aiohttp ClientSession for connection reuse"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _should_retry_responses_error(self, status: int) -> bool:
        return status in {400, 408, 409, 429} or status >= 500

    def _responses_retry_delay(self, attempt: int, retry_sleep_s: float) -> float:
        return retry_sleep_s * (2 ** attempt)

    def _responses_content_type_for_role(self, role: str) -> str:
        return "output_text" if role == "assistant" else "input_text"

    def _normalize_responses_message_content(self, role: str, content) -> list:
        if content is None:
            content = ""

        if isinstance(content, str):
            return [{
                "type": self._responses_content_type_for_role(role),
                "text": content,
            }]

        if not isinstance(content, list):
            return [{
                "type": self._responses_content_type_for_role(role),
                "text": str(content),
            }]

        normalized_blocks = []
        for block in content:
            if not isinstance(block, dict):
                normalized_blocks.append({
                    "type": self._responses_content_type_for_role(role),
                    "text": str(block),
                })
                continue

            block_type = block.get("type")
            if block_type in {
                "input_text",
                "input_image",
                "output_text",
                "refusal",
                "input_file",
                "computer_screenshot",
                "summary_text",
            }:
                normalized_blocks.append(block)
                continue

            if block_type == "text":
                normalized_blocks.append({
                    "type": self._responses_content_type_for_role(role),
                    "text": block.get("text", ""),
                })
                continue

            if block_type == "image_url":
                image_url = block.get("image_url", {})
                if isinstance(image_url, dict):
                    url = image_url.get("url", "")
                else:
                    url = image_url or ""
                normalized_blocks.append({
                    "type": "input_image",
                    "image_url": url,
                })
                continue

            normalized_blocks.append(block)

        return normalized_blocks

    async def _call_responses_api(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None,
        reasoning_effort: str = "medium",
    ) -> dict:
        """Call OpenAI's /v1/responses endpoint for GPT-5 with reasoning and tools"""

        # Convert messages from chat completions format to responses API format
        # Responses API uses different format for tool calls in conversation history
        normalized_messages = []
        for msg in messages:
            m = msg.to_dict() if hasattr(msg, 'to_dict') else dict(msg)
            role = m.get('role', '')

            # Handle assistant messages with tool_calls
            if role == 'assistant' and 'tool_calls' in m:
                # First add the assistant message content if any
                if m.get('content'):
                    normalized_messages.append({
                        "role": "assistant",
                        "content": self._normalize_responses_message_content(role, m['content']),
                    })
                # Then add each tool call as a separate function_call item
                for tc in m.get('tool_calls', []):
                    func = tc.get('function', {}) if isinstance(tc, dict) else tc.function
                    if isinstance(tc, dict):
                        call_id = tc.get('id', '')
                        func_name = func.get('name', '')
                        func_args = func.get('arguments', '')
                    else:
                        call_id = tc.id
                        func_name = func.name
                        func_args = func.arguments
                    normalized_messages.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": func_name,
                        "arguments": func_args,
                    })
            # Handle tool response messages
            elif role == 'tool':
                normalized_messages.append({
                    "type": "function_call_output",
                    "call_id": m.get('tool_call_id', ''),
                    "output": m.get('content', ''),
                })
            else:
                normalized_messages.append({
                    "role": role,
                    "content": self._normalize_responses_message_content(role, m.get('content')),
                })
        messages = normalized_messages

        if not self.base_url:
            raise ValueError(
                "base_url is required for GPT-5 responses API calls. "
                "Set sales_agent_model.base_url or export OPENAI_BASE_URL."
            )

        # Build the responses API URL
        responses_url = self.base_url.rstrip('/').replace('/v1', '') + '/v1/responses'

        headers = {
            "Content-Type": "application/json",
        }
        if self.config.get('api_key'):
            headers["Authorization"] = f"Bearer {self.config['api_key']}"
        if self.custom_api_key:
            headers["X-Api-Key"] = self.custom_api_key

        # Build the request payload for responses API
        # Note: temperature is not supported with reasoning models
        # Increase max_output_tokens since reasoning consumes tokens too
        data = {
            "model": model,
            "input": messages,
            "max_output_tokens": max(max_tokens * 2, 8192),
            "reasoning": {"effort": reasoning_effort},
        }

        # Convert tools from chat completions format to responses API format
        if tools:
            converted_tools = []
            for tool in tools:
                if tool.get("type") == "function" and "function" in tool:
                    # Convert {"type": "function", "function": {"name": ..., ...}}
                    # to {"type": "function", "name": ..., ...}
                    func = tool["function"]
                    converted_tools.append({
                        "type": "function",
                        "name": func.get("name"),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    })
                else:
                    converted_tools.append(tool)
            data["tools"] = converted_tools
        if tool_choice:
            data["tool_choice"] = tool_choice

        max_retries = int(os.environ.get("OPENAI_RESPONSES_MAX_RETRIES", os.environ.get("VLLM_MAX_RETRIES", "5")))
        retry_sleep_s = float(os.environ.get("OPENAI_RESPONSES_RETRY_SLEEP_S", os.environ.get("VLLM_RETRY_SLEEP_S", "2")))

        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(responses_url, headers=headers, json=data)
                    if response.status_code != 200:
                        error_text = response.text
                        request_id = response.headers.get("x-request-id")
                        _log_responses_api_error(
                            messages=messages,
                            model=model,
                            status=response.status_code,
                            url=str(response.url),
                            error_text=error_text,
                            request_id=request_id,
                        )
                        logger.error(f"Responses API error {response.status_code}: {error_text}")
                        if self._should_retry_responses_error(response.status_code) and attempt < max_retries:
                            delay = self._responses_retry_delay(attempt, retry_sleep_s)
                            logger.warning(
                                "Responses API transient error %s (attempt %s/%s), retrying in %.1fs...",
                                response.status_code,
                                attempt + 1,
                                max_retries + 1,
                                delay,
                            )
                            await asyncio.sleep(delay)
                            continue
                        raise Exception(f"Responses API error {response.status_code}: {error_text}")
                    result = response.json()
                    break
            except httpx.TimeoutException as e:
                if attempt < max_retries:
                    delay = self._responses_retry_delay(attempt, retry_sleep_s)
                    logger.warning(
                        "Responses API timeout (attempt %s/%s), retrying in %.1fs...",
                        attempt + 1,
                        max_retries + 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
            except httpx.ConnectError as e:
                if attempt < max_retries:
                    delay = self._responses_retry_delay(attempt, retry_sleep_s)
                    logger.warning(
                        "Responses API connection error %s (attempt %s/%s), retrying in %.1fs...",
                        type(e).__name__,
                        attempt + 1,
                        max_retries + 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

        # Convert responses API output to chat completions format
        output_items = result.get("output", [])

        # Find message content and tool calls from output
        content = None
        tool_calls = []
        reasoning = ""

        for item in output_items:
            item_type = item.get("type")
            if item_type == "reasoning":
                # Extract reasoning/thinking content
                summary = item.get("summary", [])
                if summary:
                    reasoning = "\n".join(s.get("text", "") for s in summary if s.get("type") == "summary_text")
            elif item_type == "message":
                # Extract message content
                msg_content = item.get("content", [])
                for c in msg_content:
                    if c.get("type") == "output_text":
                        content = c.get("text", "")
            elif item_type == "function_call":
                # Extract tool/function calls
                tool_calls.append({
                    "id": item.get("call_id", item.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    }
                })

        # Build a response compatible with the expected format
        message_dict = {
            "role": "assistant",
            "content": content,
        }
        if tool_calls:
            message_dict["tool_calls"] = tool_calls

        choice = {
            "index": 0,
            "finish_reason": "stop" if not tool_calls else "tool_calls",
            "message": message_dict,
        }

        return {
            'choices': VLLMResponseWrapper.wrap_choices([choice]),
            'reasoning': reasoning,
        }

    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None,
        extra_body: Optional[dict] = None,
    ) -> dict:
        """Generate a chat completion response using LiteLLM asynchronously"""
        try:
            # load model token length.
            last_message = messages[-1]["content"]
            if isinstance(last_message, list):
                last_text = " ".join(
                    block.get("text", "") for block in last_message if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                last_text = last_message or ""
            if last_text and len(last_text.split(" ")) > (max_tokens-200):
                raise Exception("Dialogue length exceeds model's maximum tokens.")

            # Check if we need to use /v1/responses API for GPT-5 with reasoning and tools
            reasoning_effort = extra_body.get("reasoning_effort") if extra_body else None
            is_gpt5 = "gpt-5" in model.lower()

            if is_gpt5 and reasoning_effort and tools:
                # Use responses API for GPT-5 with reasoning_effort and tools
                async with api_semaphore:
                    return await self._call_responses_api(
                        messages=messages,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        tools=tools,
                        tool_choice=tool_choice,
                        reasoning_effort=reasoning_effort,
                    )

            if "model_name" in self.config:
                del self.config["model_name"]
            # Build LiteLLM parameters
            llm_params = {
                'model': model,
                'messages': messages,
                'max_tokens': max_tokens,
                'temperature': temperature,
                **self.config  # Include all configuration
            }

            # Add tools if provided
            if tools:
                llm_params['tools'] = tools
            if tool_choice:
                llm_params['tool_choice'] = tool_choice
            if extra_body:
                llm_params['extra_body'] = extra_body
            async with api_semaphore:
                response = await acompletion(**llm_params)

            # Extract reasoning content if available (for models that support it)
            reasoning = ''
            if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
                reasoning = response.choices[0].message.reasoning_content

            return {
                'choices': response.choices,
                'reasoning': reasoning
            }

        except Exception as e:
            logger.error(f"LiteLLM API error: {str(e)}")
            raise Exception(f"LiteLLM API error: {str(e)}")




def create_client_from_model_name(**kwargs) -> AIClient:
    if "qwen" in kwargs["model_name"] or "glm" in kwargs["model_name"] or "gemma" in kwargs["model_name"]:
        return VLLMClient(**kwargs)
    else:
        return LiteLLMClient(**kwargs)
