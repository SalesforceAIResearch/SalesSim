#!/usr/bin/env python3
"""
Postprocesssimulation results:
1. Extract first add_to_cart(product=XYZ) or end_conversation() from shopper text
2. Put in shopper_action field (only if add_to_cart has product arg)
3. Clean shopper text by removing tool-call artifacts if there are actions in the text.
4. Discard conversation turns after this action
5. Update total_turns
"""

import json
import re
import os
import sys
from pathlib import Path

def extract_action(text):
    """Extract first add_to_cart(...) or end_conversation() from text."""

    # Patterns for add_to_cart with keyword argument (product= or product_name=)
    # Handle quoted strings properly to allow parentheses inside product names
    add_cart_kwarg_dquote = r'add_to_cart\s*\(\s*(?:product|product_name)\s*=\s*"([^"]+)"\s*\)'
    add_cart_kwarg_squote = r"add_to_cart\s*\(\s*(?:product|product_name)\s*=\s*'([^']+)'\s*\)"
    add_cart_kwarg_backtick = r'add_to_cart\s*\(\s*(?:product|product_name)\s*=\s*`([^`]+)`\s*\)'
    add_cart_kwarg_unquoted = r'add_to_cart\s*\(\s*(?:product|product_name)\s*=\s*([A-Za-z][^"\'\`\)]{2,})\s*\)'

    # Patterns for add_to_cart with positional argument
    # Handle quoted strings properly to allow parentheses inside product names
    add_cart_pos_dquote = r'add_to_cart\s*\(\s*"([^"]+)"\s*\)'
    add_cart_pos_squote = r"add_to_cart\s*\(\s*'([^']+)'\s*\)"
    add_cart_pos_backtick = r'add_to_cart\s*\(\s*`([^`]+)`\s*\)'
    add_cart_pos_unquoted = r'add_to_cart\s*\(\s*([A-Za-z][^"\'\`\)]{2,})\s*\)'

    # Patterns for <tool_call> format
    # <tool_call>add_to_cart</tool_call> - no product
    # <tool_call>add_to_cart("product")</tool_call> - with product
    tool_call_add_cart_no_arg = r'<tool_call>\s*add_to_cart\s*</tool_call>'
    tool_call_add_cart_dquote = r'<tool_call>\s*add_to_cart\s*\(\s*"([^"]+)"\s*\)\s*</tool_call>'
    tool_call_add_cart_squote = r"<tool_call>\s*add_to_cart\s*\(\s*'([^']+)'\s*\)\s*</tool_call>"
    tool_call_end_conv = r'<tool_call>\s*end_conversation\s*(?:\(\s*\))?\s*</tool_call>'

    # Gemma4 format: <|tool_call>call:add_to_cart{product:<|"|>PRODUCT<|"|>}
    gemma4_add_cart = r'<\|tool_call>call:add_to_cart\{product:<\|"\|>([^<]+)<\|"\|>\}'
    gemma4_end_conv = r'<\|tool_call>call:end_conversation\{\}'

    # Pattern for end_conversation - with or without parentheses
    # Matches: end_conversation(), end_conversation</tool_call>, `end_conversation()`
    end_conv_pattern = r'end_conversation\s*(?:\(\s*\))?'

    # Find all matches - try quoted patterns first (more reliable)
    kwarg_patterns = [add_cart_kwarg_dquote, add_cart_kwarg_squote, add_cart_kwarg_backtick, add_cart_kwarg_unquoted]
    pos_patterns = [add_cart_pos_dquote, add_cart_pos_squote, add_cart_pos_backtick, add_cart_pos_unquoted]

    kwarg_match = None
    for pattern in kwarg_patterns:
        kwarg_match = re.search(pattern, text)
        if kwarg_match:
            break

    positional_match = None
    for pattern in pos_patterns:
        positional_match = re.search(pattern, text)
        if positional_match:
            break

    # Check for <tool_call> format
    tool_call_add_no_arg_match = re.search(tool_call_add_cart_no_arg, text)
    tool_call_add_dquote_match = re.search(tool_call_add_cart_dquote, text)
    tool_call_add_squote_match = re.search(tool_call_add_cart_squote, text)
    tool_call_end_match = re.search(tool_call_end_conv, text)

    # Check for gemma4 format
    gemma4_add_match = re.search(gemma4_add_cart, text)
    gemma4_end_match = re.search(gemma4_end_conv, text)

    # Helper to extract product from the same line as the tool_call
    def extract_product_from_line(text, match_pos):
        """Extract product name from the same line as the tool_call tag."""
        # Find the line containing the match
        line_start = text.rfind('\n', 0, match_pos) + 1
        line_end = text.find('\n', match_pos)
        if line_end == -1:
            line_end = len(text)
        line = text[line_start:line_end]

        # Remove the tool_call tag itself from the line
        line_cleaned = re.sub(r'<tool_call>\s*add_to_cart\s*</tool_call>', '', line).strip()

        # Try to find quoted product names first
        quoted = re.search(r'["\']([^"\']+)["\']', line_cleaned)
        if quoted:
            return quoted.group(1).strip()

        # Try to find product name patterns like "the X" or "buy X" before punctuation
        # Look for text after "the", "buy", "purchase", "add", "get"
        product_pattern = r'(?:the|buy|purchase|add|get)\s+([A-Z][A-Za-z0-9\s\-\(\)]+?)(?:\.|,|!|\?|$)'
        product_match = re.search(product_pattern, line_cleaned, re.IGNORECASE)
        if product_match:
            return product_match.group(1).strip()

        # Fallback: return the cleaned line if it looks like a product name (starts with capital)
        if line_cleaned and line_cleaned[0].isupper() and len(line_cleaned) < 100:
            # Remove common prefixes
            line_cleaned = re.sub(r'^(I\'ll|I will|I\'d like to|Let me|Going to)\s+', '', line_cleaned, flags=re.IGNORECASE)
            if line_cleaned:
                return line_cleaned.strip().rstrip('.,!?')

        return ''

    end_conv_match = re.search(end_conv_pattern, text)

    # Collect all valid matches with their positions
    candidates = []
    if kwarg_match:
        candidates.append(('add_to_cart', kwarg_match.group(1).strip(), kwarg_match.start()))
    if positional_match:
        # Avoid matching empty or "..." style placeholders
        product = positional_match.group(1).strip()
        if product and product != '...' and not product.startswith('...'):
            candidates.append(('add_to_cart', product, positional_match.start()))

    # <tool_call> format matches
    if tool_call_add_dquote_match:
        candidates.append(('add_to_cart', tool_call_add_dquote_match.group(1).strip(), tool_call_add_dquote_match.start()))
    if tool_call_add_squote_match:
        candidates.append(('add_to_cart', tool_call_add_squote_match.group(1).strip(), tool_call_add_squote_match.start()))
    if tool_call_add_no_arg_match:
        # add_to_cart with no product specified - try to extract from same line
        product_from_line = extract_product_from_line(text, tool_call_add_no_arg_match.start())
        candidates.append(('add_to_cart', product_from_line, tool_call_add_no_arg_match.start()))
    if tool_call_end_match:
        candidates.append(('end_conversation', None, tool_call_end_match.start()))

    # Gemma4 format matches
    if gemma4_add_match:
        candidates.append(('add_to_cart', gemma4_add_match.group(1).strip(), gemma4_add_match.start()))
    if gemma4_end_match:
        candidates.append(('end_conversation', None, gemma4_end_match.start()))

    if end_conv_match:
        candidates.append(('end_conversation', None, end_conv_match.start()))

    if not candidates:
        return None, -1

    # Return the first one by position
    candidates.sort(key=lambda x: x[2])
    action_type, product, pos = candidates[0]

    if action_type == 'add_to_cart':
        # Format as JSON for grade_decision_alignment.py compatibility
        action_json = json.dumps({
            "name": "add_to_cart",
            "arguments": json.dumps({"product": product})
        })
        return action_json, pos
    else:
        return json.dumps({"name": "end_conversation", "arguments": "{}"}), pos


def clean_shopper_text(text):
    """Remove tool-call markup/function-call artifacts from shopper text."""
    cleaned = text

    # Remove XML-like tool call blocks.
    cleaned = re.sub(r'<tool_call>\s*.*?\s*</tool_call>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Remove gemma4 tool call blocks.
    cleaned = re.sub(r'<\|tool_call>call:\w+\{[^}]*\}', ' ', cleaned)

    # Remove inline function-call style artifacts (optionally wrapped in backticks).
    cleaned = re.sub(r'`?\badd_to_cart\s*\([^)]*\)\s*`?', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'`?\bend_conversation\s*(?:\(\s*\))?\s*`?', ' ', cleaned, flags=re.IGNORECASE)

    # Remove stray function names that may appear without arguments.
    cleaned = re.sub(r'\b(?:add_to_cart|end_conversation)\b', ' ', cleaned, flags=re.IGNORECASE)

    # Normalize whitespace and punctuation spacing.
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\s+([.,!?;:])', r'\1', cleaned)
    cleaned = cleaned.strip()

    return cleaned

def process_file(input_path):
    """Process a single jsonl file."""
    output_path = input_path.replace('.jsonl', '_sanitized.jsonl')

    processed_records = []
    skipped_errors = 0

    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f):
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            if record.get('outcome') == 'error':
                skipped_errors += 1
                continue

            conversation = record.get('conversation', [])
            action_found = False
            action_turn_idx = -1

            # Find first shopper turn with an action
            for idx, turn in enumerate(conversation):
                if turn.get('speaker') == 'Shopper':
                    text = turn.get('text', '')
                    action, pos = extract_action(text)

                    if action:
                        # Update shopper_action
                        turn['shopper_action'] = action
                        action_found = True
                        action_turn_idx = idx
                        cleaned_text = clean_shopper_text(text)
                        turn['text'] = cleaned_text
                        break

            # If action found, truncate conversation after this turn
            if action_found and action_turn_idx >= 0:
                record['conversation'] = conversation[:action_turn_idx + 1]
                record['total_turns'] = len(record['conversation'])

            processed_records.append(record)

    # Write output
    with open(output_path, 'w') as f:
        for record in processed_records:
            f.write(json.dumps(record) + '\n')

    print(f"Processed {input_path} -> {output_path}")
    print(f"  Total records: {len(processed_records)}")
    print(f"  Skipped (outcome=error): {skipped_errors}")

    return output_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python sanitize_tool_call_formatting.py <input_directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])

    if not input_dir.exists():
        print(f"Error: Directory '{input_dir}' does not exist")
        sys.exit(1)

    # Find all result jsonl files (exclude already processed ones)
    jsonl_files = [f for f in input_dir.glob('*_results.jsonl')
                   if '_sanitized' not in str(f)]

    print(f"Found {len(jsonl_files)} files to process")

    for filepath in jsonl_files:
        process_file(str(filepath))

if __name__ == '__main__':
    main()
