import os
import re
from typing import Union

FUZZY_MATCH_THRESHOLD = 0.7



def normalize_product_name(name: str) -> str:
    """Normalize product name for comparison: lowercase, remove punctuation, collapse whitespace."""
    name = name.lower()
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def tokenize(name: str) -> set[str]:
    """Tokenize a normalized product name."""
    return set(normalize_product_name(name).split())


def token_similarity(name1: str, name2: str) -> float:
    """Compute Jaccard similarity between two product names based on tokens."""
    tokens1 = tokenize(name1)
    tokens2 = tokenize(name2)
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)


def fuzzy_match_product(product: str, candidates: set[str], threshold: float = FUZZY_MATCH_THRESHOLD) -> str | None:
    """
    Find the best fuzzy match for a product name among candidates.
    Returns the matched candidate name if similarity >= threshold, else None.
    """
    if not product or not candidates:
        return None

    if product in candidates:
        return product

    best_match = None
    best_score = 0.0
    for candidate in candidates:
        score = token_similarity(product, candidate)
        if score > best_score:
            best_score = score
            best_match = candidate
        # Try taking the candidate first N tokens, which is usually a proper noun.
        score = token_similarity(product, " ".join(candidate.split()[:len(product.split())+2]))
        if score > best_score:
            best_score = score
            best_match = candidate
    if best_score >= threshold:
        return best_match
    return None


def subset_match_product(product: str, candidates: set[str]) -> str | None:
    """Find the best subset match for a product name among candidates."""
    if not product or not candidates:
        return None
    for candidate in candidates:
        if product.lower() in candidate.lower():
            return candidate
    return None


def _collect_cleaned_candidates(raw: str, out: set[str]) -> None:
    """Clean raw text and add viable candidate versions for catalog matching."""
    c = re.sub(r"^[\s*•]+", "", raw)
    c = re.sub(r"^\d+\.\s*", "", c).strip()
    c = re.sub(r"^\*\*(.*?)\*\*$", r"\1", c).strip()

    if not c or not re.search(r"[A-Za-z]", c):
        return

    out.add(c)


def parse_titles_from_salesperson_text(text: str) -> set[str]:
    """Recover product titles from salesperson text.

    Extracts bold text and price-line candidates, then matches each against
    the product catalog using fuzzy and subset matching.  Returns the matched
    canonical catalog title so downstream comparisons work reliably.
    """
    titles = set()
    if not text:
        return titles

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        matches = set(re.findall(r"\*\*([^*]{3,300})\*\*", stripped)) | set(re.findall(r"\d+\.\s*(.{2,300})", stripped))
        for match in matches:
            _collect_cleaned_candidates(match, titles)
  

    return titles


def find_recommended_items(candidates, response_text):
    """Find recommended items from candidates that appear in response text.

    Args:
        candidates: list of Documents or dicts with page_content/metadata
        response_text: the salesperson's response text
    Returns:
        list of dicts, each with 'page_content' and 'metadata' (including 'title')
    """
    candidate_titles = set()
    title_to_items = {}
    for item in candidates:
        if hasattr(item, 'metadata'):
            meta = item.metadata
            pc = item.page_content
        else:
            meta = item.get('metadata', {})
            pc = item.get('page_content', '')
        title = meta.get('title', '')
        if title:
            candidate_titles.add(title)
            title_to_items.setdefault(title, {'page_content': pc, 'metadata': meta})

    extracted_titles = parse_titles_from_salesperson_text(response_text)

    matched = []
    seen = set()
    for extracted in extracted_titles:
        hit = fuzzy_match_product(extracted, candidate_titles) or subset_match_product(extracted, candidate_titles)
        if hit and hit not in seen:
            seen.add(hit)
            matched.append(title_to_items[hit])
    return matched
