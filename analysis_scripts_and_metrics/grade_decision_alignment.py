#!/usr/bin/env python3
"""
Check decision alignment for conversations.

Reads a conversation JSONL file, extracts add_to_cart and recommended_items,
cross-checks with persona acceptable products, and sets
decision_alignment = True when:
  - add_to_cart is in the persona's acceptable products, OR
  - there are no recommended items in any turn AND add_to_cart is empty
Otherwise decision_alignment = False.

Usage:
  python grade_decision_alignment.py results.jsonl --path-to-acceptable-products personas.jsonl -o output.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

from salessim.simulation_utils import create_persona_hash
from salessim.agents.utils import (
    fuzzy_match_product,
    normalize_product_name,
    subset_match_product,
    parse_titles_from_salesperson_text,
)


def load_catalog_titles(catalog_path: str) -> set[str]:
    """Load all product names from the catalog JSON for a given product category."""
    if not Path(catalog_path).exists():
        print(f"Warning: catalog file not found, may affect recommended items parsing from salesperson text: {catalog_path}", file=sys.stderr)
        return set()
    with open(Path(catalog_path)) as f:
        data = json.load(f)
    data = list(data.values())[0] 
    return {p["name"] for p in data if isinstance(p, dict) and "name" in p}


def extract_item_title(item: object) -> str | None:
    """Extract a product title from a recommended_items entry."""
    if isinstance(item, str):
        title = item.strip()
        return title or None
    if not isinstance(item, dict):
        return None

    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        title = metadata.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()

    page_content = item.get("page_content")
    if isinstance(page_content, str) and page_content.strip():
        first_line = page_content.splitlines()[0].strip()
        return first_line or None
    return None


def get_salesperson_turn_text(turn: dict) -> str:
    """Return the textual content of a salesperson turn across text/content formats."""
    text = turn.get("text")
    if isinstance(text, str) and text.strip():
        return text
    if isinstance(text, list):
        text_blocks = []
        for block in text:
            if isinstance(block, dict) and block.get("type") == "text":
                block_text = block.get("text")
                if isinstance(block_text, str) and block_text.strip():
                    text_blocks.append(block_text)
        return "\n".join(text_blocks)
    return ""


def get_recommended_products(conversation: list, catalog_titles: set[str] | None = None) -> set[str]:
    """Extract all product names recommended by the salesperson."""
    products = set()
    for turn in conversation:
        if turn.get("speaker") != "Salesperson":
            continue
        if recommended := turn.get("recommended_items"):
            for item in recommended:
                title = extract_item_title(item)
                if title:
                    products.add(title)
        else:
            # NOTE: This path should be used less with improvements in salesperson scaffolding.
            # Fallback in case recommended items, see if there's recommendation-like items in text.
            text = get_salesperson_turn_text(turn)
            products.update(parse_titles_from_salesperson_text(text))
    return products


def get_add_to_cart_from_conversation(conversation: list) -> str | None:
    """Extract the product name from the first add_to_cart action in the conversation (last in reverse order)."""
    for turn in reversed(conversation):
        if turn.get("speaker") != "Shopper":
            continue
        action = turn.get("shopper_action") or ""
        if "add_to_cart" not in str(action):
            continue
        try:
            outer = json.loads(action)
            args_str = outer.get("arguments", "{}")
            if isinstance(args_str, str):
                args = json.loads(args_str)
            else:
                args = args_str or {}
            return args.get("product") or args.get("product_name")
        except (json.JSONDecodeError, TypeError):
            m = re.search(r'"product"\s*:\s*"([^"]+)"', action)
            if m:
                return m.group(1)
            m = re.search(r'"product_name"\s*:\s*"([^"]+)"', action)
            if m:
                return m.group(1)
    return None


def match_product_in_salesperson_text(
    product: str,
    conversation: list,
) -> str:
    """Match a product name against the raw text of salesperson turns.

    Scans every Salesperson turn and checks whether *product* appears via
    exact substring.
    """
    norm_product = normalize_product_name(product)

    for idx, turn in enumerate(conversation):
        if turn.get("speaker") != "Salesperson":
            continue

        text = get_salesperson_turn_text(turn)
        if not text:
            continue

        norm_text = normalize_product_name(text)

        # 1. Exact (case-insensitive) substring
        if norm_product in norm_text:
            return product

    return None


def conversation_has_recommended_acceptable_items(recommended: set[str], acceptable_items: set[str]) -> bool:
    """Return True if any Salesperson turn has recommended_items containing acceptable items."""
    if not acceptable_items:
        return False
    for product in recommended:
        if product in acceptable_items:
            return True
        fuzzy_match = fuzzy_match_product(product, acceptable_items)
        if fuzzy_match and fuzzy_match in acceptable_items:
            return True
        # subset match using subset_match_product
        subset_match = subset_match_product(product, acceptable_items)
        if subset_match and subset_match in acceptable_items:
            return True
    return False


def build_persona_hash_to_acceptable_products(personas_jsonl_path: Path) -> tuple[dict[str, set[str]], dict[tuple[str, int], set[str]]]:
    """
    Build mapping persona_hash -> set(acceptable product names).
    Also builds a fallback mapping (name, age) -> set(acceptable product names).
    Reads persona JSONL: hash from persona fields, acceptable products from
    acceptable_products and ideal_products on each line.

    Returns:
        tuple: (hash_to_acceptable, name_age_to_acceptable)
    """
    hash_to_acceptable = {}
    name_age_to_acceptable = {}
    with open(personas_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            shopper_persona = {
                "name": p.get("name"),
                "age": p.get("age"),
                "background": p.get("persona_background", ""),
            }
            shopper_preferences = f"preferences: {p.get('preferences', '')}\ndealbreakers: {p.get('dealbreakers', '')}"
            h = create_persona_hash(shopper_persona, shopper_preferences)
            # Handle both string and dict formats for acceptable/ideal products
            def extract_names(items):
                if not items:
                    return []
                return [item["name"] if isinstance(item, dict) else item for item in items]
            acceptable = set(extract_names(p.get("acceptable_products")))
            hash_to_acceptable[h] = acceptable

            # Also store by (name, age) as fallback
            name = p.get("name")
            age = p.get("age")
            if name is not None and age is not None:
                name_age_to_acceptable[(name, age)] = acceptable

    return hash_to_acceptable, name_age_to_acceptable


def compute_decision_alignment_with_outcome_category(
    record: dict,
    hash_to_acceptable: dict[str, set[str]],
    catalog_titles: set[str] | None = None,
    name_age_to_acceptable: dict[tuple[str, int], set[str]] | None = None,
) -> tuple[bool, str]:
    """
    Returns a tuple (decision_alignment, outcome_category) where outcome_category is one of:
    - premature_ending
    - overly_lenient_no_recommended_acceptable
    - shopperbot_hallucination_no_credit
    - chose_wrong_recommended_product
    - correct_reasoning_accepted_recommended_product_acceptable
    - correct_reasoning_reject_no_recommended_acceptable
    - incorrect_reasoning_reject_has_recommended_acceptable
    """
    conversation = record.get("conversation") or record.get("messages") or []
    add_to_cart_product = get_add_to_cart_from_conversation(conversation)
    # Try to get acceptable products by hash first
    gt_acceptable = hash_to_acceptable.get(record.get("persona_hash"), set())

    # If not found and we have name_age_to_acceptable, try fallback
    if not gt_acceptable and name_age_to_acceptable is not None:
        shopper_persona = record.get("shopper_persona", {})
        name = shopper_persona.get("name")
        age = shopper_persona.get("age")
        if name is not None and age is not None:
            gt_acceptable = name_age_to_acceptable.get((name, age), set())
    recommended = get_recommended_products(conversation, catalog_titles)
    has_recommended_acceptable = conversation_has_recommended_acceptable_items(recommended, gt_acceptable)


    # Premature ending (only greeting, no recommendations)
    premature_ending = record.get("total_turns") == 1
    if premature_ending:
        return False, "premature_ending"
    if add_to_cart_product: # Shopperbot purchased, check if the accepted product is in the acceptable list.
        if not gt_acceptable:
            # no acceptable products, overly lenient. 
            # NOTE: We don't check for hallucinations from salesperson. 
            return False, "overly_lenient_no_recommended_acceptable"
        matched_fuzzy_recommended = fuzzy_match_product(add_to_cart_product, recommended)
        matched_subset = subset_match_product(add_to_cart_product, recommended) or match_product_in_salesperson_text(add_to_cart_product, conversation)
        if not matched_fuzzy_recommended and not matched_subset:
            # Shopperbot hallucination: product that was accepted was never recommended
            # error. 
            return False, "shopperbot_hallucination_no_credit"

        # now match if the matched recommended product is in the acceptable list.
        matched_fuzzy_acceptable = fuzzy_match_product(add_to_cart_product, gt_acceptable)
        matched_subset_acceptable = subset_match_product(add_to_cart_product, gt_acceptable)
        # Check if the matched recommended product (that the salesperson recommended), matched by 
        # fuzzy match or subset match, is in the acceptable list.
        matched_acceptable = matched_fuzzy_acceptable in gt_acceptable or matched_subset_acceptable in gt_acceptable
        if not matched_acceptable and has_recommended_acceptable:
            return False, "chose_wrong_recommended_product"
        elif not matched_acceptable and not has_recommended_acceptable:
            return False, "overly_lenient_no_recommended_acceptable"
        return True, "correct_reasoning_accepted_recommended_product_acceptable"
    else:
        # No purchase: aligned if no acceptable items were recommended
        # If purchase, model is too stringent. 
        if not has_recommended_acceptable:
            return True, "correct_reasoning_reject_no_recommended_acceptable"
        else:
            return False, "incorrect_reasoning_reject_has_recommended_acceptable"



def main():
    correct_outcome_categories = ["correct_reasoning_accepted_recommended_product_acceptable", "correct_reasoning_reject_no_recommended_acceptable"]
    incorrect_outcome_categories = ["premature_ending", "overly_lenient_no_recommended_acceptable", "shopperbot_hallucination_no_credit", "chose_wrong_recommended_product", "incorrect_reasoning_reject_has_recommended_acceptable"]
    parser = argparse.ArgumentParser(description="Check decision alignment for conversations")
    parser.add_argument(
        "conversation_to_grade",
        type=Path,
        help="Path to results.jsonl (or similar conversation JSONL)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: stdout)",
    )
    parser.add_argument(
        "--path-to-acceptable-products",
        type=Path,
        required=True,
        help="Path to acceptable_products.jsonl (used for hash and acceptable products)",
    )
    parser.add_argument(
        "--path-to-catalog",
        type=str,
        required=True,
        help="Path to catalog directory (e.g. data/products). "
             "Used to load catalog titles for fallback title extraction from salesperson text.",
    )
    args = parser.parse_args()

    if not args.path_to_acceptable_products.exists():
        print(f"Error: path to acceptable products file not found: {args.path_to_acceptable_products}", file=sys.stderr)
        sys.exit(1)
    if not args.conversation_to_grade.exists():
        print(f"Error: conversation to grade file not found: {args.conversation_to_grade}", file=sys.stderr)
        sys.exit(1)

    hash_to_acceptable, name_age_to_acceptable = build_persona_hash_to_acceptable_products(args.path_to_acceptable_products)
    catalog_titles = load_catalog_titles(args.path_to_catalog) if args.path_to_catalog else None
    total = 0
    aligned = 0
    skipped = 0
    summary = {"incorrect_outcome": 0, "correct_outcome": 0,
                "incorrect_outcome_categories": {category: 0 for category in incorrect_outcome_categories},
                "correct_outcome_categories": {category: 0 for category in correct_outcome_categories}}

    out_file = open(args.output, "w") if args.output else sys.stdout
    try:
        with open(args.conversation_to_grade) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                # Skip records with missing persona hash
                persona_hash = record.get("persona_hash")
                name = record.get("shopper_persona", {}).get("name")
                age = record.get("shopper_persona", {}).get("age")
                if persona_hash not in hash_to_acceptable and (name, age) not in name_age_to_acceptable:
                    skipped += 1
                    print(f"Skipping record with missing persona_hash: {persona_hash}", file=sys.stderr)
                    continue

                decision_alignment, outcome_category = compute_decision_alignment_with_outcome_category(record, hash_to_acceptable, catalog_titles, name_age_to_acceptable)

                # Add decision_alignment and outcome_category to the record
                record["decision_alignment"] = decision_alignment
                record["outcome_category"] = outcome_category

                if decision_alignment:
                    summary["correct_outcome"] += 1
                    summary["correct_outcome_categories"][outcome_category] += 1
                    aligned += 1
                else:
                    summary["incorrect_outcome"] += 1
                    summary["incorrect_outcome_categories"][outcome_category] += 1
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
    finally:
        if args.output:
            out_file.close()

    alignment_rate = (aligned / total) if total else 0.0
    print("\n=== Decision Alignment Summary ===", file=sys.stderr)
    print(f"Alignment rate: {alignment_rate:.1%}", file=sys.stderr)
    print(f"Total graded: {total}", file=sys.stderr)
    print(f"Aligned: {aligned}", file=sys.stderr)
    if skipped > 0:
        print(f"Skipped (missing persona_hash): {skipped}", file=sys.stderr)
    print("\n  Correct behavior categories breakdown:", file=sys.stderr)
    for cat, val in summary["correct_outcome_categories"].items():
        print(f"    {cat}: {val}", file=sys.stderr)
    print("\n  Incorrect behavior categories breakdown:", file=sys.stderr)
    for cat, val in summary["incorrect_outcome_categories"].items():
        print(f"    {cat}: {val}", file=sys.stderr)
    print("\n=================================\n", file=sys.stderr)

    if args.output:
        print(f"Wrote results with decision_alignment to {args.output}", file=sys.stderr)

    summary["skipped_missing_persona_hash"] = skipped
    print(json.dumps(summary, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
