"""Gemma 4 model prompt formatting."""

from typing import List, Dict
from common import register_prompt_formatting
from data.products.domain_knowledge import DOMAIN_KNOWLEDGE
from .gemma import (
    base_template_short,
    base_template_human_steering,
)

GEMMA4_TOOL_CALL = '<|tool_call>call:add_to_cart{{product:<|"|>{product}<|"|>}} '

RECOMMENDATION_EXAMPLE_BASE = """- You may ask for your preferences, but you will settle for products that fulfill only your dealbreakers. Put your reasoning in <reasoning></reasoning> tags, before outputting the response as the shopper back to the salesperson."""

RECOMMENDATION_EXAMPLES = {
    "smart_watch": """ Here's an example. <reasoning>Preferences: I prefer a smartwatch with GPS features. Dealbreakers: I cannot exceed $100. Let me check each of the products step by step, first with dealbreakers then preferences. Let us start with the Mas watch. The only dealbreaker is that it is $100. Since the watch is $120, this is over the budget and disqualifies the watch. Let us consider the Destiny watch. The only dealbreaker is that it is $100. Since the watch is $90, this is in budget. The only preference is for GPS, which the Destiny watch does not have. However, since it fulfills all dealbreakers I will still add to cart. Since GPS features are a preference and it satisfies all dealbreakers, it is acceptable for Alex.</reasoning>Great! I will take it the Destiny.
""" + GEMMA4_TOOL_CALL.format(product="Destiny Watch") + """

Here's another example. <reasoning>Preferences: I prefer GPS tracking, heart rate monitoring, and a sport style. Dealbreakers: I cannot exceed $200 and must be water resistant. Let me check each of the products step by step, first with dealbreakers then preferences. Let us start with the FitPro Watch. The dealbreakers are budget under $200 and water resistant. Since the FitPro Watch costs $180 and is water resistant, it satisfies all dealbreakers. Now checking preferences: it has GPS tracking (1 preference met), no heart rate monitoring (0), and sport style (1 preference met). Total: 2 out of 3 preferences. Let us consider the ActiveBand Watch. The dealbreakers are budget under $200 and water resistant. Since the ActiveBand Watch costs $150 and is water resistant, it satisfies all dealbreakers. Now checking preferences: it has GPS tracking (1 preference met), heart rate monitoring (1 preference met), and sport style (1 preference met). Total: 3 out of 3 preferences. Both products satisfy all dealbreakers, but the ActiveBand Watch fulfills more preferences (3 vs 2), so I will choose the ActiveBand Watch.</reasoning>Great! I will take the ActiveBand Watch.
""" + GEMMA4_TOOL_CALL.format(product="ActiveBand Watch") + """
""",
    "laptops": """ Here's an example. <reasoning>Preferences: I prefer a laptop with 16GB memory and a lightweight design. Dealbreakers: I cannot exceed $1,500 budget and must have an SSD. Let me check each of the products step by step, first with dealbreakers then preferences. Let us start with the Dell XPS. The dealbreakers are budget under $1,500 and SSD required. Since the Dell XPS costs $1,799, this is over budget and disqualifies the laptop. Let us consider the HP Pavilion. The dealbreakers are budget under $1,500 and SSD required. Since the HP Pavilion costs $899 and has a 256GB SSD, it satisfies all dealbreakers. The preference is for 16GB memory, but the HP Pavilion has 8GB memory. However, since it fulfills all dealbreakers I will still add to cart. Since 16GB memory is a preference and it satisfies all dealbreakers, it is acceptable.</reasoning>Great! I will take the HP Pavilion.
""" + GEMMA4_TOOL_CALL.format(product="HP Pavilion") + """

Here's another example. <reasoning>Preferences: I prefer 16GB memory, a touchscreen display, and backlit keyboard. Dealbreakers: I cannot exceed $1,200 budget and must have an SSD. Let me check each of the products step by step, first with dealbreakers then preferences. Let us start with the Lenovo IdeaPad. The dealbreakers are budget under $1,200 and SSD required. Since the Lenovo IdeaPad costs $799 and has a 512GB SSD, it satisfies all dealbreakers. Now checking preferences: it has 8GB memory (0 preferences met), no touchscreen (0), and backlit keyboard (1 preference met). Total: 1 out of 3 preferences. Let us consider the ASUS VivoBook. The dealbreakers are budget under $1,200 and SSD required. Since the ASUS VivoBook costs $949 and has a 256GB SSD, it satisfies all dealbreakers. Now checking preferences: it has 16GB memory (1 preference met), touchscreen display (1 preference met), and backlit keyboard (1 preference met). Total: 3 out of 3 preferences. Both products satisfy all dealbreakers, but the ASUS VivoBook fulfills more preferences (3 vs 1), so I will choose the ASUS VivoBook.</reasoning>Great! I will take the ASUS VivoBook.
""" + GEMMA4_TOOL_CALL.format(product="ASUS VivoBook") + """
""",
    "female_clothing": """ Here's an example. <reasoning>Preferences: I prefer 100% cotton material and neutral colors. Dealbreakers: I cannot exceed $80 budget and must be machine washable. Let me check each of the products step by step, first with dealbreakers then preferences. Let us start with the Silk Blouse. The dealbreakers are budget under $80 and machine washable. Since the Silk Blouse costs $120, this is over budget and disqualifies it. Let us consider the BDG Cardigan. The dealbreakers are budget under $80 and machine washable. Since the BDG Cardigan costs $59 and is machine washable, it satisfies all dealbreakers. The preference is for 100% cotton, but the BDG Cardigan is a cotton blend. However, since it fulfills all dealbreakers I will still add to cart. Since 100% cotton is a preference and it satisfies all dealbreakers, it is acceptable.</reasoning>Great! I will take the BDG Cardigan.
""" + GEMMA4_TOOL_CALL.format(product="BDG Cardigan") + """

Here's another example. <reasoning>Preferences: I prefer 100% cotton, neutral colors, and a relaxed fit. Dealbreakers: I cannot exceed $70 budget and must be machine washable. Let me check each of the products step by step, first with dealbreakers then preferences. Let us start with the Madewell Tee. The dealbreakers are budget under $70 and machine washable. Since the Madewell Tee costs $45 and is machine washable, it satisfies all dealbreakers. Now checking preferences: it is 100% cotton (1 preference met), white color which is neutral (1 preference met), and slim fit not relaxed (0). Total: 2 out of 3 preferences. Let us consider the Everlane Cotton Top. The dealbreakers are budget under $70 and machine washable. Since the Everlane Cotton Top costs $55 and is machine washable, it satisfies all dealbreakers. Now checking preferences: it is 100% cotton (1 preference met), beige color which is neutral (1 preference met), and relaxed fit (1 preference met). Total: 3 out of 3 preferences. Both products satisfy all dealbreakers, but the Everlane Cotton Top fulfills more preferences (3 vs 2), so I will choose the Everlane Cotton Top.</reasoning>Great! I will take the Everlane Cotton Top.
""" + GEMMA4_TOOL_CALL.format(product="Everlane Cotton Top") + """
""",
    "male_clothing": """ Here's an example. <reasoning>Preferences: I prefer solid colors and slim fit style. Dealbreakers: I cannot exceed $100 budget and must be 100% cotton. Let me check each of the products step by step, first with dealbreakers then preferences. Let us start with the Ralph Lauren Shirt. The dealbreakers are budget under $100 and 100% cotton. Since the Ralph Lauren Shirt costs $168, this is over budget and disqualifies it. Let us consider the Barbour Shirt. The dealbreakers are budget under $100 and 100% cotton. Since the Barbour Shirt costs $85 and is 100% cotton, it satisfies all dealbreakers. The preference is for solid colors, but the Barbour Shirt has a plaid pattern. However, since it fulfills all dealbreakers I will still add to cart. Since solid color is a preference and it satisfies all dealbreakers, it is acceptable.</reasoning>Great! I will take the Barbour Shirt.
""" + GEMMA4_TOOL_CALL.format(product="Barbour Shirt") + """

Here's another example. <reasoning>Preferences: I prefer solid colors, slim fit, and button-down collar. Dealbreakers: I cannot exceed $90 budget and must be 100% cotton. Let me check each of the products step by step, first with dealbreakers then preferences. Let us start with the J.Crew Oxford Shirt. The dealbreakers are budget under $90 and 100% cotton. Since the J.Crew Oxford Shirt costs $79 and is 100% cotton, it satisfies all dealbreakers. Now checking preferences: it is striped not solid (0 preferences met), slim fit (1 preference met), and button-down collar (1 preference met). Total: 2 out of 3 preferences. Let us consider the Bonobos Everyday Shirt. The dealbreakers are budget under $90 and 100% cotton. Since the Bonobos Everyday Shirt costs $88 and is 100% cotton, it satisfies all dealbreakers. Now checking preferences: it is solid navy color (1 preference met), slim fit (1 preference met), and button-down collar (1 preference met). Total: 3 out of 3 preferences. Both products satisfy all dealbreakers, but the Bonobos Everyday Shirt fulfills more preferences (3 vs 2), so I will choose the Bonobos Everyday Shirt.</reasoning>Great! I will take the Bonobos Everyday Shirt.
""" + GEMMA4_TOOL_CALL.format(product="Bonobos Everyday Shirt") + """
""",
    "skincare": """ Here's an example. <reasoning>Preferences: I prefer products with SPF protection and anti-aging ingredients. Dealbreakers: I cannot exceed $50 budget and must be fragrance-free. Let me check each of the products step by step, first with dealbreakers then preferences. Let us start with the Premium Serum Set. The dealbreakers are budget under $50 and fragrance-free. Since the Premium Serum Set costs $85, this is over budget and disqualifies it. Let us consider the Olay Regenerist Lotion. The dealbreakers are budget under $50 and fragrance-free. Since the Olay Regenerist Lotion costs $28 and is fragrance-free, it satisfies all dealbreakers. The preference is for SPF protection, but the Olay Regenerist Lotion does not include SPF. However, since it fulfills all dealbreakers I will still add to cart. Since SPF is a preference and it satisfies all dealbreakers, it is acceptable.</reasoning>Great! I will take the Olay Regenerist Lotion.
""" + GEMMA4_TOOL_CALL.format(product="Olay Regenerist Lotion") + """

Here's another example. <reasoning>Preferences: I prefer SPF protection, anti-aging ingredients, and lightweight texture. Dealbreakers: I cannot exceed $45 budget and must be fragrance-free. Let me check each of the products step by step, first with dealbreakers then preferences. Let us start with the Neutrogena Hydro Boost. The dealbreakers are budget under $45 and fragrance-free. Since the Neutrogena Hydro Boost costs $20 and is fragrance-free, it satisfies all dealbreakers. Now checking preferences: it has no SPF (0 preferences met), contains hyaluronic acid for hydration but no anti-aging retinol (0), and lightweight gel texture (1 preference met). Total: 1 out of 3 preferences. Let us consider the CeraVe AM Facial Moisturizing Lotion. The dealbreakers are budget under $45 and fragrance-free. Since the CeraVe AM Lotion costs $17 and is fragrance-free, it satisfies all dealbreakers. Now checking preferences: it has SPF 30 (1 preference met), contains niacinamide for anti-aging (1 preference met), and lightweight texture (1 preference met). Total: 3 out of 3 preferences. Both products satisfy all dealbreakers, but the CeraVe AM Lotion fulfills more preferences (3 vs 1), so I will choose the CeraVe AM Lotion.</reasoning>Great! I will take the CeraVe AM Facial Moisturizing Lotion.
""" + GEMMA4_TOOL_CALL.format(product="CeraVe AM Facial Moisturizing Lotion") + """
""",
    "beauty": """ Here's an example. <reasoning>Preferences: I prefer matte finish options and buildable coverage. Dealbreakers: I cannot exceed $30 budget and must be long-lasting formula. Let me check each of the products step by step, first with dealbreakers then preferences. Let us start with the Tarte Palette. The dealbreakers are budget under $30 and long-lasting formula. Since the Tarte Palette costs $45, this is over budget and disqualifies it. Let us consider the Revlon Highlighting Palette. The dealbreakers are budget under $30 and long-lasting formula. Since the Revlon Highlighting Palette costs $15 and has a long-lasting formula, it satisfies all dealbreakers. The preference is for matte finish, but the Revlon Highlighting Palette only offers shimmer finishes. However, since it fulfills all dealbreakers I will still add to cart. Since matte finish is a preference and it satisfies all dealbreakers, it is acceptable.</reasoning>Great! I will take the Revlon Highlighting Palette.
""" + GEMMA4_TOOL_CALL.format(product="Revlon Highlighting Palette") + """

Here's another example. <reasoning>Preferences: I prefer matte finish, buildable coverage, and a variety of neutral shades. Dealbreakers: I cannot exceed $25 budget and must be long-lasting formula. Let me check each of the products step by step, first with dealbreakers then preferences. Let us start with the e.l.f. Mad for Matte Palette. The dealbreakers are budget under $25 and long-lasting formula. Since the e.l.f. Palette costs $10 and has a long-lasting formula, it satisfies all dealbreakers. Now checking preferences: it has matte finish (1 preference met), sheer coverage not buildable (0), and limited neutral shades (0). Total: 1 out of 3 preferences. Let us consider the NYX Ultimate Shadow Palette. The dealbreakers are budget under $25 and long-lasting formula. Since the NYX Palette costs $18 and has a long-lasting formula, it satisfies all dealbreakers. Now checking preferences: it has matte finish options (1 preference met), buildable coverage (1 preference met), and wide variety of neutral shades (1 preference met). Total: 3 out of 3 preferences. Both products satisfy all dealbreakers, but the NYX Ultimate Shadow Palette fulfills more preferences (3 vs 1), so I will choose the NYX Ultimate Shadow Palette.</reasoning>Great! I will take the NYX Ultimate Shadow Palette.
""" + GEMMA4_TOOL_CALL.format(product="NYX Ultimate Shadow Palette") + """
""",
    "game_gadgets": """ Here's an example. <reasoning>Preferences: I prefer games with expansion packs and multiplayer support for 4+ players. Dealbreakers: I cannot exceed $50 budget and must be suitable for ages 8+. Let me check each of the products step by step, first with dealbreakers then preferences. Let us start with the Collector's Edition Board Game. The dealbreakers are budget under $50 and suitable for ages 8+. Since the Collector's Edition Board Game costs $90, this is over budget and disqualifies it. Let us consider the Jenga Giant Game. The dealbreakers are budget under $50 and suitable for ages 8+. Since the Jenga Giant Game costs $40 and is suitable for ages 8+, it satisfies all dealbreakers. The preference is for expansion packs, but the Jenga Giant Game does not include any expansions. However, since it fulfills all dealbreakers I will still add to cart. Since expansion packs are a preference and it satisfies all dealbreakers, it is acceptable.</reasoning>Great! I will take the Jenga Giant Game.
""" + GEMMA4_TOOL_CALL.format(product="Jenga Giant Game") + """

Here's another example. <reasoning>Preferences: I prefer expansion packs available, multiplayer for 4+ players, and quick setup time. Dealbreakers: I cannot exceed $45 budget and must be suitable for ages 8+. Let me check each of the products step by step, first with dealbreakers then preferences. Let us start with the Ticket to Ride. The dealbreakers are budget under $45 and suitable for ages 8+. Since Ticket to Ride costs $40 and is suitable for ages 8+, it satisfies all dealbreakers. Now checking preferences: it has expansion packs available (1 preference met), supports 2-5 players so 4+ works (1 preference met), but has longer setup time (0). Total: 2 out of 3 preferences. Let us consider the Catan Board Game. The dealbreakers are budget under $45 and suitable for ages 8+. Since Catan costs $44 and is suitable for ages 10+ which covers 8+, it satisfies all dealbreakers. Now checking preferences: it has many expansion packs (1 preference met), supports 3-4 players so 4+ works (1 preference met), and has quick setup time (1 preference met). Total: 3 out of 3 preferences. Both products satisfy all dealbreakers, but Catan fulfills more preferences (3 vs 2), so I will choose Catan.</reasoning>Great! I will take the Catan Board Game.
""" + GEMMA4_TOOL_CALL.format(product="Catan Board Game") + """
""",
}

RECOMMENDATION_EXAMPLE_DEFAULT = """ Here's an example. <reasoning>Preferences: I prefer premium features and extended warranty. Dealbreakers: I cannot exceed my budget and must meet minimum quality standards. Let me check each of the products step by step, first with dealbreakers then preferences. Let us start with the Premium Option. The dealbreakers are staying within budget and minimum quality. Since the Premium Option exceeds my budget, this disqualifies it. Let us consider the Standard Option. The dealbreakers are staying within budget and minimum quality. Since the Standard Option is within budget and meets quality standards, it satisfies all dealbreakers. The preference is for premium features, but the Standard Option has basic features. However, since it fulfills all dealbreakers I will still add to cart. Since premium features are a preference and it satisfies all dealbreakers, it is acceptable.</reasoning>Great! I will take the Standard Option.
""" + GEMMA4_TOOL_CALL.format(product="Standard Option") + """

Here's another example. <reasoning>Preferences: I prefer premium features, extended warranty, and fast shipping. Dealbreakers: I cannot exceed my budget and must meet minimum quality standards. Let me check each of the products step by step, first with dealbreakers then preferences. Let us start with Option A. The dealbreakers are staying within budget and minimum quality. Since Option A is within budget and meets quality standards, it satisfies all dealbreakers. Now checking preferences: it has basic features not premium (0 preferences met), no extended warranty (0), and standard shipping (0). Total: 0 out of 3 preferences. Let us consider Option B. The dealbreakers are staying within budget and minimum quality. Since Option B is within budget and meets quality standards, it satisfies all dealbreakers. Now checking preferences: it has premium features (1 preference met), extended warranty included (1 preference met), and fast shipping available (1 preference met). Total: 3 out of 3 preferences. Both products satisfy all dealbreakers, but Option B fulfills more preferences (3 vs 0), so I will choose Option B.</reasoning>Great! I will take Option B.
""" + GEMMA4_TOOL_CALL.format(product="Option B") + """
"""


def get_recommendation_example(product_category: str | None = None) -> str:
    """Get the category-specific recommendation example for Gemma 4."""
    if product_category and product_category in RECOMMENDATION_EXAMPLES:
        return RECOMMENDATION_EXAMPLE_BASE + RECOMMENDATION_EXAMPLES[product_category]
    return RECOMMENDATION_EXAMPLE_BASE + RECOMMENDATION_EXAMPLE_DEFAULT


def _create_gemma4_prompt(curr_content, curr_preferences: str, persona: str, chat_history: List[Dict], product_category: str | None = None, is_recommendation=False):
    domain_knowledge = DOMAIN_KNOWLEDGE.get(product_category)
    recommendation_example = get_recommendation_example(product_category) if is_recommendation else ""
    system_prompt = base_template_short.format(
        preferences=curr_preferences,
        persona=persona,
        recommendation_example=recommendation_example,
        domain_knowledge=domain_knowledge
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        if msg["speaker"] == "Salesperson":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["speaker"] == "Shopper":
            messages.append({"role": "assistant", "content": msg["content"]})
    messages.append({"role": "user", "content": curr_content})
    return messages


def _create_gemma4_prompt_human_steering(curr_content, curr_preferences: str, persona: str, chat_history: List[Dict], product_category: str | None = None, is_recommendation=False):
    domain_knowledge = DOMAIN_KNOWLEDGE.get(product_category)
    recommendation_example = get_recommendation_example(product_category) if is_recommendation else ""
    system_prompt = base_template_human_steering.format(
        preferences=curr_preferences,
        persona=persona,
        recommendation_example=recommendation_example,
        domain_knowledge=domain_knowledge
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        if msg["speaker"] == "Salesperson":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["speaker"] == "Shopper":
            messages.append({"role": "assistant", "content": msg["content"]})
    messages.append({"role": "user", "content": curr_content})
    return messages


@register_prompt_formatting("hosted_vllm/gemma4_with_reasoning")
def create_gemma4_prompt(curr_content, curr_preferences: str, persona: str, chat_history: List[Dict], product_category: str | None = None, is_recommendation=False):
    return _create_gemma4_prompt(curr_content, curr_preferences, persona, chat_history, product_category=product_category, is_recommendation=is_recommendation)


@register_prompt_formatting("hosted_vllm/gemma4_with_reasoning_human_steering")
def create_gemma4_prompt_human_steering(curr_content, curr_preferences: str, persona: str, chat_history: List[Dict], product_category: str | None = None, is_recommendation=False):
    return _create_gemma4_prompt_human_steering(curr_content, curr_preferences, persona, chat_history, product_category=product_category, is_recommendation=is_recommendation)


@register_prompt_formatting("hosted_vllm/gemma4_with_thinking")
def create_gemma4_thinking_prompt(curr_content, curr_preferences: str, persona: str, chat_history: List[Dict], product_category: str | None = None, is_recommendation=False):
    return _create_gemma4_prompt(curr_content, curr_preferences, persona, chat_history, product_category=product_category, is_recommendation=is_recommendation)


@register_prompt_formatting("hosted_vllm/gemma4_with_thinking_human_steering")
def create_gemma4_thinking_prompt_human_steering(curr_content, curr_preferences: str, persona: str, chat_history: List[Dict], product_category: str | None = None, is_recommendation=False):
    return _create_gemma4_prompt_human_steering(curr_content, curr_preferences, persona, chat_history, product_category=product_category, is_recommendation=is_recommendation)
