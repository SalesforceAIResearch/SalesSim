#!/usr/bin/env python3
"""Check tool-call formatting issues in results JSONL.

Usage:
    python3 analysis_scripts_and_metrics/calculate_toolcall_error_metric.py \
      --input simulations/glm46_with_reasoning/male_clothing/male_clothing_results.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


import re

TOOL_CALL_FORMATS = [
    {"name": "hermes", "open": "<tool_call>", "close": "</tool_call>"},
    {"name": "gemma4", "open": "<|tool_call>", "close": "tool_call|>"},
]

# Sort longest-first so substring tags are checked before their prefixes
_ALL_OPEN_TAGS = sorted(
    [fmt["open"] for fmt in TOOL_CALL_FORMATS], key=len, reverse=True
)

_lookbehinds = "".join(
    f"(?<!{re.escape(tag)})" for tag in _ALL_OPEN_TAGS
)
UNTAGGED_TOOL_CALL_PATTERN = re.compile(
    _lookbehinds + r'\b(add_to_cart|end_conversation)\s*\([^)]*\)',
    re.IGNORECASE,
)


@dataclass
class Issue:
    issue_type: str
    conversation_id: str
    line_number: int
    turn: int | None
    speaker: str | None
    detail: str
    text_excerpt: str


def short_excerpt(text: str, max_len: int = 220) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3] + "..."


def load_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            rows.append((line_no, json.loads(stripped)))
    return rows


def analyze_turn_text(
    conversation_id: str, line_no: int, turn_obj: dict[str, Any]
) -> list[Issue]:
    issues: list[Issue] = []

    # Handle text that might be a list
    raw_text = turn_obj.get("text", "")
    if isinstance(raw_text, list) and len(raw_text) > 0:
        text = str(raw_text[0].get("text", "") if isinstance(raw_text[0], dict) else raw_text[0])
    else:
        text = str(raw_text or "")

    speaker = turn_obj.get("speaker")
    turn = turn_obj.get("turn")

    tag_counts: dict[str, dict[str, int]] = {}
    for fmt in TOOL_CALL_FORMATS:
        tag_counts[fmt["name"]] = {
            "open": text.count(fmt["open"]),
            "close": text.count(fmt["close"]),
        }

    # Deduplicate substring overlaps: longer tags that contain shorter ones
    # e.g. "<|tool_call>" contains "<tool_call>"
    for i, longer in enumerate(TOOL_CALL_FORMATS):
        for shorter in TOOL_CALL_FORMATS[i + 1:]:
            if shorter["open"] in longer["open"]:
                tag_counts[shorter["name"]]["open"] -= tag_counts[longer["name"]]["open"]
            if shorter["close"] in longer["close"]:
                tag_counts[shorter["name"]]["close"] -= tag_counts[longer["name"]]["close"]

    has_any_tool_markup = any(
        counts["open"] > 0 or counts["close"] > 0
        for counts in tag_counts.values()
    )
    if has_any_tool_markup:
        detail_parts = []
        for name, counts in tag_counts.items():
            if counts["open"] > 0 or counts["close"] > 0:
                detail_parts.append(f"{name}: open={counts['open']}, close={counts['close']}")
        issues.append(
            Issue(
                issue_type="tool_call_in_text_with_tags",
                conversation_id=conversation_id,
                line_number=line_no,
                turn=turn,
                speaker=speaker,
                detail="; ".join(detail_parts),
                text_excerpt=short_excerpt(text),
            )
        )

    # Check for untagged tool calls (add_to_cart/end_conversation without <tool_call> tags)
    untagged_matches = UNTAGGED_TOOL_CALL_PATTERN.findall(text)
    if untagged_matches and not has_any_tool_markup:
        issues.append(
            Issue(
                issue_type="tool_call_in_text_without_tags",
                conversation_id=conversation_id,
                line_number=line_no,
                turn=turn,
                speaker=speaker,
                detail=f"untagged tool calls: {len(untagged_matches)}",
                text_excerpt=short_excerpt(text),
            )
        )

    return issues


def check_tool_formatting_issues(rows: list[tuple[int, dict[str, Any]]]) -> int:
    conversation_has_issues = 0

    for line_no, obj in rows:
        conversation_id = str(obj.get("conversation_id", f"line_{line_no}"))
        turns = obj.get("conversation")

        for turn_obj in turns:
            turn_issues = analyze_turn_text(conversation_id, line_no, turn_obj)
            if len(turn_issues) > 0:
                conversation_has_issues += 1
                break

    return conversation_has_issues

def count_first_shopper_toolcall_instances(
    rows: list[tuple[int, dict[str, Any]]],
) -> tuple[int, int]:
    """Count tool-call markup in first shopper turns.

    Returns:
        (raw_count, unique_conversation_count)
    """
    raw_count = 0

    for _, obj in rows:
        turns = obj.get("conversation")
        if not isinstance(turns, list) or not turns:
            continue
        for  turn_obj in turns:
            speaker = str(turn_obj.get("speaker", "")).lower()
            if speaker == "shopper":
                if "add_to_cart" in turn_obj["shopper_action"] or "end_conversation" in turn_obj["shopper_action"]:
                    raw_count += 1
                break
                

    return raw_count


def write_json_report(path: Path, issues: list[Issue], per_issue_type: Counter[str]) -> None:
    payload = {
        "total_issues": len(issues),
        "issue_type_counts": dict(per_issue_type),
        "issues": [issue.__dict__ for issue in issues],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def print_console_summary(
    total_rows: int,
    issues: list[Issue],
    per_issue_type: Counter[str],
    per_conv_issue_count: Counter[str],
    first_shopper_toolcall_raw_count: int,
    first_shopper_toolcall_unique_count: int,
    sample_limit: int,
) -> None:
    impacted_conversations = len({i.conversation_id for i in issues})
    print(f"Total conversations: {total_rows}")
    print(f"Conversations with tool-call issues: {impacted_conversations}")
    print(f"Total tool-call issues: {len(issues)}")
    print(
        "First shopper turns with tool-call markup: "
        f"{first_shopper_toolcall_raw_count} raw, "
        f"{first_shopper_toolcall_unique_count} unique conversations"
    )
    print("\nIssue counts by type:")
    for issue_type, count in per_issue_type.most_common():
        print(f"  - {issue_type}: {count}")

    if not issues:
        return

    print("\nMost impacted conversations:")
    for cid, count in per_conv_issue_count.most_common(10):
        print(f"  - {cid}: {count} issues")

    print("\nSample issues:")
    shown = 0
    for issue in issues:
        if shown >= sample_limit:
            break
        print(
            "  - "
            f"[{issue.issue_type}] cid={issue.conversation_id} line={issue.line_number} "
            f"turn={issue.turn} speaker={issue.speaker} :: {issue.detail}"
        )
        if issue.text_excerpt:
            print(f"      excerpt: {issue.text_excerpt}")
        shown += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check results JSONL for improperly formatted tool calls."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to results JSONL (e.g., male_clothing_results.jsonl).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = load_jsonl(input_path)
    conversation_has_issues_count = check_tool_formatting_issues(rows)
    first_shopper_toolcall_raw_count = (
        count_first_shopper_toolcall_instances(rows)
    )
    print(f"% of conversations with premature tool call endings: {first_shopper_toolcall_raw_count / len(rows) * 100}%")
    print(f"% of conversations with tool call formatting issues: {conversation_has_issues_count / len(rows) * 100}%")
if __name__ == "__main__":
    main()
