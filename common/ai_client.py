from abc import ABC, abstractmethod
from typing import List, Dict
import os
import asyncio
from litellm import acompletion
# Semaphore to limit concurrent API calls (used in batch simulations)
MAX_CONCURRENT_API_CALLS = 5
api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)


class AIClient(ABC):
    """Abstract base class for AI clients"""

    def __init__(self):
        self.total_cost = 0.0


    @abstractmethod
    async def async_chat_completion(self, messages: List[Dict[str, str]], model: str, max_tokens: int, temperature: float, tools: List[dict] = None, tool_choice: str = None) -> dict:
        """Generate a chat completion response asynchronously

        Returns:
            dict: {
                'choices': str,    # The actual response content
                'reasoning': str,  # The reasoning/thinking process (empty string if not available)
            }
        """
        pass

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
        if base_url:
            self.config['base_url'] = base_url

        # Custom headers handling
        headers = {}
        if extra_headers:
            headers.update(extra_headers)

        # Custom API key for special authentication
        if custom_api_key:
            headers["X-Api-Key"] = custom_api_key
        elif custom_api_key_env:
            custom_key_env_name = custom_api_key_env
            custom_key = os.environ.get(custom_key_env_name)
            if custom_key:
                headers["X-Api-Key"] = custom_key

        if headers:
            self.config['extra_headers'] = headers

        # Store additional kwargs for special model configurations
        for key, value in kwargs.items():
            self.config[key] = value

    async def async_chat_completion(self, messages: List[Dict[str, str]], model: str, max_tokens: int, temperature: float, tools: List[dict] = None, tool_choice: str = None) -> dict:
        """Generate a chat completion response using LiteLLM asynchronously"""
        try:
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
            raise Exception(f"LiteLLM API error: {str(e)}")




def create_client_from_model_name(**kwargs) -> AIClient:
    return LiteLLMClient(**kwargs)
