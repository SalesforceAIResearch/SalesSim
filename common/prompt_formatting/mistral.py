"""Mistral model prompt formatting."""

from typing import List
from common import register_prompt_formatting


base_template = """You are shopping online for a {product_label} at a store, and are communicating with a digital salesperson via a chat interface to learn more about the store's offerings to make an informed decision.

Here are some details about yourself.
{persona}

Follow these rules:
- Chat with the salesperson to learn more about products in this category. They will be acting as a product expert, helping you make an informed purchasing decisions as well as helping you understand the options available in the store.
- Use your preferences and incorporate them in your responses when appropriate in a way that is in line with your persona. This often means not reveal them to the salesperson right away or all at once.
- When the salesperson makes a recommendation, please consider whether the product satisfies your assigned preferences and dealbreakers, and decide to either buy or keep looking based on that.
{recommendation_example}\n- If you would like to accept or buy a product, call the add_to_cart function with the product name as the parameter.
- If the recommended product is not a good fit, let the salesperson know (e.g. "this is too expensive")
- If you're not sure about the recommended product, feel free to ask follow-up questions (e.g. "could you explain the benefit of this feature?")
- Feel free to end the conversation at any time by calling the end_conversation function. You MUST call this function to end the conversation.
- Embody the persona and personality traits in your responses completely in your communication and behavior.

Your preferences and dealbreakers:
{preferences}

For budget preferences, unless it is a dealbreaker, you should be willing to accept up to 10% over the budget. REMEMBER: Generate utterances in the style of someone who is writing in an e-commerce store chatbot interface online, not role-playing.
Follow the above rules to generate a reply."""

mistral_system_template = base_template + """
Follow the above rules to generate a reply using your assigned preferences and the conversation history:"""


def _format_product_label(product_category: str | None) -> str:
    if not product_category:
        return "product"
    if product_category == "laptops":
        return "laptop"
    return product_category.replace("_", " ")


@register_prompt_formatting("mistral")
def create_mistral_prompt(curr_preferences: str, persona: str, big5_prompt: str, chat_history: List[str], input_txt: str, product_category: str | None = None):
    product_label = _format_product_label(product_category)
    system_content = mistral_system_template.format(
        preferences=curr_preferences,
        persona=persona,
        product_label=product_label
    )

    messages = [{"role": "system", "content": system_content}]

    # Add chat history as alternating user/assistant messages
    for _, msg in enumerate(chat_history):
        if msg.startswith("Salesperson: "):
            content = msg.replace("Salesperson: ", "")
            messages.append({"role": "user", "content": content})
        elif msg.startswith("Shopper: "):
            content = msg.replace("Shopper: ", "")
            messages.append({"role": "assistant", "content": content})

    # Add current input as user message
    messages.append({"role": "user", "content": input_txt})
    return messages
