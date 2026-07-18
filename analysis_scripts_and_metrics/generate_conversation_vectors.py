#!/usr/bin/env python3
"""Analyze SalesSim dialogues and compute various metrics."""

import argparse
import asyncio
import json
import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.asyncio import tqdm_asyncio


MODEL = "gpt-4o-mini"
client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

# Semaphore to limit concurrent requests
MAX_CONCURRENT = 50

def get_category_from_results_file_name(name: str) -> str:
    """Get category from results file name."""
    return name.replace("_results_postprocess.jsonl", "")

def load_dialogues(dialogue_dir: str) -> dict:
    """Load all dialogue JSONL files from the directory."""
    dialogues = {}
    for filepath in glob.glob(os.path.join(dialogue_dir, "*_postprocess.jsonl")):

        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    dialogue = json.loads(line)
                    dialogue_id = dialogue.get("conversation_id", "")
                    dialogue["_category"] = get_category_from_results_file_name(filepath)
                    dialogues[dialogue_id] = dialogue

    return dialogues


def get_shopper_messages(dialogue: dict) -> list[str]:
    """Extract all shopper messages from dialogue."""
    return [
        turn["text"]
        for turn in dialogue.get("conversation", [])
        if turn.get("speaker") == "Shopper"
    ]


async def count_first_turn_preferences_llm(first_message: str, semaphore: asyncio.Semaphore) -> int:
    """Use LLM to count number of revealed preferences in the first turn."""
    if not first_message:
        return 0

    prompt = f"""Analyze this shopper's first message in a product recommendation conversation.
    Count the number of distinct preferences, requirements, or criteria they mention.

    Examples of preferences:
    - Budget constraints (e.g., "under $500")
    - Physical attributes (e.g., "lightweight", "compact")
    - Use case requirements (e.g., "for commuting", "for gaming")
    - Feature requirements (e.g., "long battery life", "waterproof")
    - Performance needs (e.g., "fast", "high resolution")
    - Style preferences (e.g., "modern look", "black color")

    Message:
    "{first_message}"

    Respond with ONLY a single integer representing the count of distinct preferences. Nothing else."""

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip()
            return int(result)
        except Exception as e:
            return 0


async def calculate_grammatical_completeness_llm(messages: list[str], semaphore: asyncio.Semaphore) -> float:
    """Use LLM to calculate percentage of grammatically complete sentences."""
    if not messages:
        return 0.0

    all_text = " ".join(messages)
    if not all_text.strip():
        return 0.0

    prompt = f"""Analyze the following text from a shopper in a conversation.
Calculate what percentage of the sentences are grammatically complete and correct.

A grammatically complete sentence:
- Has a subject and predicate
- Uses proper punctuation
- Has correct grammar and syntax

Text:
"{all_text}"

Respond with ONLY a decimal number between 0.0 and 1.0 representing the fraction of grammatically complete sentences. Nothing else."""

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip()
            return float(result)
        except Exception as e:
            return 0.0


def count_turns(dialogue: dict) -> int:
    """Count number of shopper turns in conversation."""
    return len(get_shopper_messages(dialogue))


def calculate_tfidf_similarities(dialogues: dict) -> dict:
    """Calculate TF-IDF cosine similarities within each product category."""
    category_dialogues = defaultdict(list)
    for dialogue_id, dialogue in dialogues.items():
        category = dialogue["_category"]
        shopper_text = " ".join(get_shopper_messages(dialogue))
        category_dialogues[category].append((dialogue_id, shopper_text))

    similarities = {}

    for category, items in category_dialogues.items():
        if len(items) < 2:
            for dialogue_id, _ in items:
                similarities[dialogue_id] = {
                    "avg_tfidf_similarity": 0.0,
                    "p75_tfidf_similarity": 0.0,
                    "p25_tfidf_similarity": 0.0,
                }
            continue

        dialogue_ids = [item[0] for item in items]
        texts = [item[1] for item in items]

        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            for dialogue_id in dialogue_ids:
                similarities[dialogue_id] = {
                    "avg_tfidf_similarity": 0.0,
                    "p75_tfidf_similarity": 0.0,
                    "p25_tfidf_similarity": 0.0,
                }
            continue

        cos_sim_matrix = cosine_similarity(tfidf_matrix)

        for i, dialogue_id in enumerate(dialogue_ids):
            sims = [cos_sim_matrix[i, j] for j in range(len(dialogue_ids)) if i != j]

            if sims:
                similarities[dialogue_id] = {
                    "avg_tfidf_similarity": float(np.mean(sims)),
                    "p75_tfidf_similarity": float(np.percentile(sims, 75)),
                    "p25_tfidf_similarity": float(np.percentile(sims, 25)),
                }
            else:
                similarities[dialogue_id] = {
                    "avg_tfidf_similarity": 0.0,
                    "p75_tfidf_similarity": 0.0,
                    "p25_tfidf_similarity": 0.0,
                }

    return similarities


def compute_per_turn_tfidf(dialogues: dict, max_turns: int = 20) -> dict:
    """Compute per-turn TF-IDF similarity within each product category."""
    # Group shopper utterances by (category, turn index).
    category_turn_messages = defaultdict(list)
    for dialogue_id, dialogue in dialogues.items():
        category = dialogue["_category"]
        shopper_messages = get_shopper_messages(dialogue)
        for turn_idx, msg in enumerate(shopper_messages):
            if turn_idx < max_turns:
                category_turn_messages[(category, turn_idx)].append((dialogue_id, msg))

    per_conversation_results = defaultdict(dict)

    for (_, turn_idx), id_msg_pairs in category_turn_messages.items():
        dialogue_ids = [dialogue_id for dialogue_id, _ in id_msg_pairs]
        messages = [msg for _, msg in id_msg_pairs]
        n = len(messages)

        if n < 2:
            # No peers to compare against for this turn.
                print(f"Not enough messages to compute TF-IDF similarity for turn {turn_idx} in category {category}")
                continue
        try:
            vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
            tfidf_matrix = vectorizer.fit_transform(messages)
            cos_sim = cosine_similarity(tfidf_matrix)
        except ValueError:
            for dialogue_id in dialogue_ids:
                per_conversation_results[dialogue_id][turn_idx] = {
                    "avg_similarity": 0.0,
                    "peer_count": n - 1,
                }
            continue

        for i, dialogue_id in enumerate(dialogue_ids):
            sims = [cos_sim[i, j] for j in range(n) if i != j]
            avg_sim = float(np.mean(sims)) if sims else 0.0
            per_conversation_results[dialogue_id][turn_idx] = {
                "avg_similarity": round(avg_sim, 4),
                "peer_count": len(sims),
            }

    return dict(per_conversation_results)


def aggregate_by_turn(results: dict, max_turns: int = 20) -> dict:
    """Aggregate per-conversation turn similarities into turn-level summaries."""
    turn_data = defaultdict(lambda: {"sims": [], "peer_counts": []})

    for _, turns in results.items():
        for turn_idx, data in turns.items():
            if turn_idx < max_turns:
                turn_data[turn_idx]["sims"].append(data["avg_similarity"])
                turn_data[turn_idx]["peer_counts"].append(data["peer_count"])

    aggregated = {}
    for turn_idx in range(max_turns):
        sims = turn_data[turn_idx]["sims"]
        peer_counts = turn_data[turn_idx]["peer_counts"]
        if sims:
            weights = [max(c, 1) for c in peer_counts]
            weighted_avg = sum(s * w for s, w in zip(sims, weights)) / sum(weights)
            aggregated[turn_idx] = {
                "avg_similarity": round(weighted_avg, 4),
                "num_conversations": len(sims),
                "avg_peer_count": round(float(np.mean(peer_counts)), 2) if peer_counts else 0.0,
            }

    return aggregated


async def process_dialogue(dialogue_id: str, dialogue: dict, tfidf_sims: dict, semaphore: asyncio.Semaphore) -> tuple:
    """Process a single dialogue and return its features."""
    shopper_messages = get_shopper_messages(dialogue)
    first_message = shopper_messages[0] if shopper_messages else ""
    pacing, completeness = await asyncio.gather(
        count_first_turn_preferences_llm(first_message, semaphore),
        calculate_grammatical_completeness_llm(shopper_messages, semaphore)
    )

    features = {
        "category": dialogue.get("_category", "unknown"),
        "criterias_mentioned_first_turn": pacing,
        "complete_sentences": round(completeness, 4),
        "num_turns": count_turns(dialogue),
        "avg_tfidf_similarity": round(tfidf_sims[dialogue_id]["avg_tfidf_similarity"], 4),
    }

    return dialogue_id, features


def generate_analysis_report(features: dict) -> str:
    """Generate a summary analysis of the features."""
    categories = defaultdict(list)
    all_pacing = []
    all_completeness = []
    all_turns = []
    all_avg_sim = []

    for dialogue_id, f in features.items():
        category = f["category"]
        categories[category].append(f)
        all_pacing.append(f["pacing"])
        all_completeness.append(f["complete_sentences"])
        all_turns.append(f["num_turns"])
        all_avg_sim.append(f["avg_tfidf_similarity"])

    report = []
    report.append("=" * 60)
    report.append("SalesSim Dialogue Feature Analysis")
    report.append("(qwen_baseline_reasoning)")
    report.append("=" * 60)
    report.append("")
    report.append(f"Total dialogues analyzed: {len(features)}")
    report.append(f"Product categories: {', '.join(sorted(categories.keys()))}")
    report.append("")

    report.append("-" * 40)
    report.append("OVERALL STATISTICS")
    report.append("-" * 40)
    report.append(f"Pacing (preferences in 1st turn): mean={np.mean(all_pacing):.2f}, median={np.median(all_pacing):.1f}, min={min(all_pacing)}, max={max(all_pacing)}")
    report.append(f"Complete sentences: mean={np.mean(all_completeness):.2%}, median={np.median(all_completeness):.2%}")
    report.append(f"Number of turns: mean={np.mean(all_turns):.1f}, median={np.median(all_turns):.1f}, min={min(all_turns)}, max={max(all_turns)}")
    report.append(f"Avg TF-IDF similarity: mean={np.mean(all_avg_sim):.4f}, median={np.median(all_avg_sim):.4f}")
    report.append("")

    report.append("-" * 40)
    report.append("PER-CATEGORY STATISTICS")
    report.append("-" * 40)

    for category in sorted(categories.keys()):
        category_features = categories[category]
        n = len(category_features)

        c_pacing = [f["pacing"] for f in category_features]
        c_completeness = [f["complete_sentences"] for f in category_features]
        c_turns = [f["num_turns"] for f in category_features]
        c_avg_sim = [f["avg_tfidf_similarity"] for f in category_features]

        report.append(f"\n{category.upper()} (n={n})")
        report.append(f"  Pacing: mean={np.mean(c_pacing):.2f}, median={np.median(c_pacing):.1f}")
        report.append(f"  Complete sentences: mean={np.mean(c_completeness):.2%}")
        report.append(f"  Turns: mean={np.mean(c_turns):.1f}, median={np.median(c_turns):.1f}")
        report.append(f"  Avg TF-IDF similarity: {np.mean(c_avg_sim):.4f}")

    report.append("")
    report.append("=" * 60)

    return "\n".join(report)


async def main(dialogue_dir: str):
    dialogue_dir = Path(dialogue_dir)

    # Load dialogues once
    dialogues = load_dialogues(str(dialogue_dir))
    print(f"Loaded {len(dialogues)} dialogues")

    # Compute per-dialogue features
    print("Calculating TF-IDF similarities...")
    tfidf_sims = calculate_tfidf_similarities(dialogues)

    # Compute per-turn TF-IDF
    print("Computing TF-IDF similarity by turn...")
    per_turn_results = compute_per_turn_tfidf(dialogues, max_turns=4)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        process_dialogue(dialogue_id, dialogue, tfidf_sims, semaphore)
        for dialogue_id, dialogue in dialogues.items()
    ]

    print(f"Processing {len(tasks)} dialogues with LLM calls...")
    results = await tqdm_asyncio.gather(*tasks, desc="Analyzing dialogues")

    features = {dialogue_id: feat for dialogue_id, feat in results}

    # Compute per-turn TF-IDF and merge as dialogue-level features.
    print("\nComputing TF-IDF similarity by turn...")
    per_turn_results = compute_per_turn_tfidf(dialogues)
    for dialogue_id, turn_map in per_turn_results.items():
        if dialogue_id not in features:
            continue
        for turn_idx, turn_data in turn_map.items():
            features[dialogue_id][f"tfidf_turn_{turn_idx}"] = turn_data["avg_similarity"]

    output_path = dialogue_dir / "dialogue_features.json"
    with open(output_path, "w") as f:
        json.dump(features, f, indent=2)
    print(f"\nFeatures saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SalesSim dialogues and compute various metrics")
    parser.add_argument(
        "dialogue_dir",
        help="top-level folder for evaluation runs for all products, feature vectors will be calculated for all product categories"
    )
    args = parser.parse_args()

    asyncio.run(main(args.dialogue_dir))
