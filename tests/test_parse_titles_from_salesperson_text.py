import unittest

from customersim.agents.utils import parse_titles_from_salesperson_text


class TestParseTitlesFromSalespersonText(unittest.TestCase):

    def test_extracts_bold_product_names(self):
        text = "I recommend the **iPhone 15 Pro Max** for your needs."
        result = parse_titles_from_salesperson_text(text)
        self.assertEqual(result, {"iPhone 15 Pro Max"})

    def test_extracts_multiple_bold_products(self):
        text = "Check out **iPhone 15 Pro Max** or **Samsung Galaxy S24 Ultra**."
        result = parse_titles_from_salesperson_text(text)
        self.assertEqual(result, {"iPhone 15 Pro Max", "Samsung Galaxy S24 Ultra"})

    def test_extracts_numbered_list_products(self):
        text = """Here are my recommendations:
1. iPhone 15 Pro Max
2. Samsung Galaxy S24 Ultra
3. Google Pixel 8 Pro"""
        result = parse_titles_from_salesperson_text(text)
        self.assertEqual(result, {"iPhone 15 Pro Max", "Samsung Galaxy S24 Ultra", "Google Pixel 8 Pro"})

    def test_extracts_mixed_bold_and_numbered(self):
        text = """Consider these options:
1. **iPhone 15 Pro Max** - Great camera
2. Samsung Galaxy S24 Ultra - Excellent display"""
        result = parse_titles_from_salesperson_text(text)
        self.assertIn("iPhone 15 Pro Max", result)
        self.assertIn("Samsung Galaxy S24 Ultra - Excellent display", result)

    def test_extracts_product_with_bullet_points(self):
        text = """I recommend:
* **iPhone 15 Pro Max**
• **Samsung Galaxy S24 Ultra**"""
        result = parse_titles_from_salesperson_text(text)
        self.assertEqual(result, {"iPhone 15 Pro Max", "Samsung Galaxy S24 Ultra"})

    def test_realistic_salesperson_response_1(self):
        text = """Based on your requirements, I'd recommend the **iPhone 15 Pro Max**.
It has an excellent camera system and great battery life.
Price: $1,199"""
        result = parse_titles_from_salesperson_text(text)
        self.assertEqual(result, {"iPhone 15 Pro Max"})

    def test_realistic_salesperson_response_2(self):
        text = """Great question! Here are my top picks:

1. **MacBook Pro 16-inch** - Best for video editing ($2,499)
2. **Dell XPS 15** - Great value alternative ($1,799)

Both have excellent displays and powerful processors."""
        result = parse_titles_from_salesperson_text(text)
        self.assertIn("MacBook Pro 16-inch", result)
        self.assertIn("Dell XPS 15", result)

    def test_realistic_salesperson_response_3(self):
        text = """For noise cancellation, you can't beat the **Sony WH-1000XM5 Headphones**.
They're industry-leading and very comfortable for long listening sessions."""
        result = parse_titles_from_salesperson_text(text)
        self.assertEqual(result, {"Sony WH-1000XM5 Headphones"})

    def test_ignores_plain_text_without_formatting(self):
        text = "I think you should consider the iPhone or Samsung phone."
        result = parse_titles_from_salesperson_text(text)
        self.assertEqual(result, set())

    def test_ignores_incomplete_markdown(self):
        text = "Check out the *iPhone 15 Pro Max or **Samsung Galaxy"
        result = parse_titles_from_salesperson_text(text)
        self.assertEqual(result, set())

    def test_ignores_very_short_bold_matches(self):
        text = "**Hi** is a short word."
        result = parse_titles_from_salesperson_text(text)
        self.assertEqual(result, set())

    def test_extracts_only_products_meeting_min_length(self):
        text = "I recommend the **iPhone 15 Pro Max** today."
        result = parse_titles_from_salesperson_text(text)
        self.assertIn("iPhone 15 Pro Max", result)

    def test_short_bold_text_not_extracted_when_isolated(self):
        text = "**OK**"
        result = parse_titles_from_salesperson_text(text)
        self.assertEqual(result, set())

    def test_ignores_numbers_only(self):
        text = "1. **123** 2. **456**"
        result = parse_titles_from_salesperson_text(text)
        self.assertEqual(result, set())

    def test_ignores_special_characters_only(self):
        text = "**!!!** and **$$$**"
        result = parse_titles_from_salesperson_text(text)
        self.assertEqual(result, set())

    def test_noise_with_random_asterisks(self):
        text = "This * is * some * random * text * with * asterisks"
        result = parse_titles_from_salesperson_text(text)
        self.assertEqual(result, set())

    def test_noise_with_partial_formatting(self):
        text = "**Product1 and Product2** but neither are real products"
        result = parse_titles_from_salesperson_text(text)
        self.assertIn("Product1 and Product2", result)

    def test_empty_text_returns_empty_set(self):
        result = parse_titles_from_salesperson_text("")
        self.assertEqual(result, set())

    def test_none_text_returns_empty_set(self):
        result = parse_titles_from_salesperson_text(None)
        self.assertEqual(result, set())

    def test_whitespace_only_text_returns_empty_set(self):
        text = "   \n\n   \t  "
        result = parse_titles_from_salesperson_text(text)
        self.assertEqual(result, set())

    def test_strips_leading_bullets_and_numbers(self):
        text = "1. **MacBook Pro 16-inch**"
        result = parse_titles_from_salesperson_text(text)
        self.assertIn("MacBook Pro 16-inch", result)

    def test_handles_nested_bold_in_numbered_list(self):
        text = """Top recommendations:
1. **Apple Watch Series 9** (best for iOS)
2. Samsung Galaxy Watch (best for Android)"""
        result = parse_titles_from_salesperson_text(text)
        self.assertIn("Apple Watch Series 9", result)
        self.assertIn("Samsung Galaxy Watch (best for Android)", result)

    def test_ignores_conversational_noise(self):
        text = """Hello! How are you today? I'm here to help you find the perfect product.
What are you looking for? Let me know your budget and preferences."""
        result = parse_titles_from_salesperson_text(text)
        self.assertEqual(result, set())

    def test_extracts_from_complex_formatting(self):
        text = """**Great choice!** Here are some options:

1. **iPhone 15 Pro Max** - $1,199
   - 256GB storage
   - A17 Pro chip

2. **Samsung Galaxy S24 Ultra** - $1,299
   - S Pen included
   - 200MP camera

Let me know which one interests you!"""
        result = parse_titles_from_salesperson_text(text)
        self.assertIn("iPhone 15 Pro Max", result)
        self.assertIn("Samsung Galaxy S24 Ultra", result)
        self.assertIn("Great choice!", result)

    def test_handles_long_text_with_price_descriptions(self):
        text = "1. Sony WH-1000XM5 Headphones - Premium noise cancellation at $399.99"
        result = parse_titles_from_salesperson_text(text)
        self.assertIn("Sony WH-1000XM5 Headphones - Premium noise cancellation at $399.99", result)

    def test_filters_text_without_letters(self):
        text = "**123 456 789**"
        result = parse_titles_from_salesperson_text(text)
        self.assertEqual(result, set())

    def test_extracts_products_with_parentheses(self):
        text = "1. **Bose QuietComfort Earbuds** (2nd Generation)"
        result = parse_titles_from_salesperson_text(text)
        self.assertIn("Bose QuietComfort Earbuds", result)

    def test_salesperson_with_pricing_info(self):
        text = """Check out these great deals:
1. iPhone 15 Pro - $999 (20% off!)
2. Samsung Galaxy S24 - $899"""
        result = parse_titles_from_salesperson_text(text)
        self.assertIn("iPhone 15 Pro - $999 (20% off!)", result)
        self.assertIn("Samsung Galaxy S24 - $899", result)

    def test_salesperson_with_emojis_and_formatting(self):
        text = """Perfect choices! ✨
1. **MacBook Pro** - Amazing performance
2. **iPad Pro** - Great for creativity"""
        result = parse_titles_from_salesperson_text(text)
        self.assertIn("MacBook Pro", result)
        self.assertIn("iPad Pro", result)

    def test_noise_with_code_or_technical_text(self):
        text = "const product = **fetchProduct()** from database"
        result = parse_titles_from_salesperson_text(text)
        self.assertIn("fetchProduct()", result)

    def test_noise_with_markdown_headers(self):
        text = "## Product Recommendations\nLet me help you"
        result = parse_titles_from_salesperson_text(text)
        self.assertEqual(result, set())

    def test_salesperson_multiline_recommendation(self):
        text = """Based on your needs, I recommend:

1. **Apple Watch Series 9**
   - Health tracking
   - Water resistant
   - $399

Would you like to know more?"""
        result = parse_titles_from_salesperson_text(text)
        self.assertIn("Apple Watch Series 9", result)

    def test_extracts_product_from_comparison(self):
        text = "1. **iPhone 15 Pro** vs **Samsung Galaxy S24**"
        result = parse_titles_from_salesperson_text(text)
        self.assertIn("iPhone 15 Pro", result)

    def test_salesperson_with_questions(self):
        text = """What are you looking for?
**Need help?** I can assist with your purchase.
Let me know your budget!"""
        result = parse_titles_from_salesperson_text(text)
        self.assertIn("Need help?", result)


if __name__ == "__main__":
    unittest.main()
