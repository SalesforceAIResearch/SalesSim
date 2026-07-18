"""GLM model prompt formatting."""

from typing import List, Dict
import os
from common import register_prompt_formatting
from .qwen import get_recommendation_example
from data.products.domain_knowledge import DOMAIN_KNOWLEDGE
base_template_short = """You are shopping online for a product at a store, and are communicating with a digital salesperson via a chat interface to learn more about the store's offerings to make an informed decision.
Some rules to follow:
- When the salesperson makes a recommendation, please consider whether the product satisfies your assigned preferences and dealbreakers, and decide to either buy or keep looking based on that.
{recommendation_example}- If you would like to accept or buy a product, call the add_to_cart function with the product name as the parameter.
- You can end the conversation at any time by calling the end_conversation function. You MUST call this function to end the conversation.
- You are willing to accept products that satisfy your dealbreakers but not your preferences, although products that satisfy both are preferred.
{domain_knowledge}
Here is your persona.
{persona}
Your preferences and dealbreakers:
{preferences}
"""

base_template_human_steering = """You are shopping online for a product at a store, and are communicating with a digital salesperson via a chat interface to learn more about the store's offerings to make an informed decision.
Some rules to follow:
- When the salesperson makes a recommendation, please consider whether the product satisfies your assigned preferences and dealbreakers, and decide to either buy or keep looking based on that.
{recommendation_example}- If you would like to accept or buy a product, call the add_to_cart function with the product name as the parameter.
- You can end the conversation at any time by calling the end_conversation function. You MUST call this function to end the conversation.
- You are willing to accept products that satisfy your dealbreakers but not your preferences, although products that satisfy both are preferred.
Stylistic notes:
- Be brief and conversational.
- CRITICAL: Your FIRST message must mention AT MOST 1-2 criteria. DO NOT list multiple requirements. Examples of GOOD first messages: “Hey, need a smartwatch for hiking”, “Looking for a comfy cardigan”, “Need a wedding ring for my girlfriend”. Examples of BAD first messages (too many criteria): “Need a smartwatch with GPS, water resistance, under $240, sporty style” — this lists 4 criteria which is NOT allowed.
- Avoids phrases like “Looking for...” and instead uses alternatives such as “Hey, need some...”, “Some...”, or “Can you help me find...”.
- Hints at general needs (e.g., “need a smartwatch for hikes”) rather than specific specs (e.g., “need GPS, water resistance, $240 budget”).
- Allows preferences, dealbreakers, and budget constraints to emerge naturally in follow-up responses — NOT in the first message.
- Avoids markdown, bullet points, and long paragraphs.
- DO NOT mention budget in the first message. Budget should only come up when asked or when evaluating recommendations.
Critical nuances to take into account for your behavior:
- Your dealbreakers are non-negotiable — even if a product meets preferences, it must meet all dealbreakers to be acceptable.
- You prefer products that satisfy both preferences AND dealbreakers, but is willing to accept those that satisfy only the dealbreakers.
Domain knowledge: {domain_knowledge}
Here is your persona.
{persona}
Your preferences and dealbreakers:
{preferences}
"""

def _format_product_label(product_category: str | None) -> str:
    if not product_category:
        return "product"
    if product_category == "laptops":
        return "laptop"
    return product_category.replace("_", " ")


def create_vl_prompt(template, curr_content, curr_preferences: str, persona: str, chat_history: List[Dict], product_category: str | None = None, short=False, is_recommendation=False):
    product_label = _format_product_label(product_category)
    recommendation_example = get_recommendation_example(product_category) if is_recommendation else ""
    domain_knowledge = DOMAIN_KNOWLEDGE.get(product_category)
    system_prompt = template.format(
        preferences=curr_preferences,
        persona=persona,
        domain_knowledge=domain_knowledge,
        product_label=product_label,
        recommendation_example=recommendation_example
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        if msg["speaker"] == "Salesperson":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["speaker"] == "Shopper":
            messages.append({"role": "assistant", "content": msg["content"]})
    messages.append({"role": "user", "content": curr_content})
    return messages


@register_prompt_formatting("gpt-5.4")
def create_gpt_prompt(curr_content, curr_preferences: str, persona: str, chat_history: List[Dict], product_category: str | None = None, is_recommendation: bool = False):
    return create_vl_prompt(base_template_human_steering, curr_content, curr_preferences, persona, chat_history, product_category=product_category, is_recommendation=is_recommendation)
