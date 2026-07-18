from typing import Callable


PROMPT_FORMATTING_REGISTRY = {}
def register_prompt_formatting(model_name: str) -> Callable[[Callable], Callable]:
    def decorator(func):
        PROMPT_FORMATTING_REGISTRY[model_name] = func
        return func
    return decorator

# Import the prompt_formatting package to register all formatters
from common import prompt_formatting
