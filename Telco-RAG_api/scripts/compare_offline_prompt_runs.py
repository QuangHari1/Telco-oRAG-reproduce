#!/usr/bin/env python3
"""Compare strict Option-N and upstream-reference results on the final records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTION_PATTERN = re.compile(r"option\D*(\d+)", re.IGNORECASE)


def latest_completed(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if isinstance(row.get("correct"), bool):
                    rows[int(row["index"])] = row
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"Invalid completed row at {path}:{line_number}") from error
    return rows


def strict_prediction(row: dict[str, Any]) -> str | None:
    value = row.get("predicted")
    if not isinstance(value, str):
        return None
    match = OPTION_PATTERN.search(value)
    return f"Option {match.group(1)}" if match else None


def reference_prediction(row: dict[str, Any]) -> str | None:
    values = row.get("predicted_option_numbers")
    if isinstance(values, list) and len(values) == 1 and str(values[0]).isdigit():
        return f"Option {values[0]}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--last", type=int, default=200)
    parser.add_argument(
        "--strict-results",
        type=Path,
        default=PROJECT_ROOT / "results" / "paper_offline_option_only_gpt-4o-mini_teleqna.jsonl",
    )
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=PROJECT_ROOT / "results" / "paper_offline_reference_prompt_gpt-4o-mini_last200.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "paper_offline_prompt_comparison_last200.json",
    )
    args = parser.parse_args()
    if args.last <= 0:
        parser.error("--last must be positive")

    strict = latest_completed(args.strict_results)
    reference = latest_completed(args.reference_results)
    common = sorted(set(strict) & set(reference))
    if len(common) < args.last:
        missing = args.last - len(common)
        raise RuntimeError(f"Only {len(common)} shared completed rows; {missing} more reference rows are required.")
    indices = common[-args.last:]
    strict_correct = sum(strict[index]["correct"] for index in indices)
    reference_correct = sum(reference[index]["correct"] for index in indices)
    both_correct = sum(strict[index]["correct"] and reference[index]["correct"] for index in indices)
    strict_only = sum(strict[index]["correct"] and not reference[index]["correct"] for index in indices)
    reference_only = sum(not strict[index]["correct"] and reference[index]["correct"] for index in indices)
    neither = args.last - both_correct - strict_only - reference_only
    comparable_predictions = [
        index for index in indices if strict_prediction(strict[index]) is not None and reference_prediction(reference[index]) is not None
    ]
    agreement = sum(strict_prediction(strict[index]) == reference_prediction(reference[index]) for index in comparable_predictions)
    summary = {
        "indices": {"start": indices[0], "end": indices[-1], "count": len(indices)},
        "strict_option_only": {"correct": strict_correct, "accuracy": strict_correct / args.last},
        "upstream_reference_prompt": {"correct": reference_correct, "accuracy": reference_correct / args.last},
        "paired_correctness": {
            "both_correct": both_correct,
            "strict_only": strict_only,
            "reference_only": reference_only,
            "both_wrong": neither,
        },
        "prediction_agreement": {
            "comparable_rows": len(comparable_predictions),
            "same_option": agreement,
            "rate": agreement / len(comparable_predictions) if comparable_predictions else None,
        },
        "strict_results": str(args.strict_results),
        "reference_results": str(args.reference_results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
