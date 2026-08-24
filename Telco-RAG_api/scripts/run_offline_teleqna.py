#!/usr/bin/env python3
"""Resume-safe evaluation of the paper's offline Telco-RAG path on TeleQnA.

The per-question call sequence is intentionally the one in
``Telco-RAG_paper_version/pipeline_offline.py``: question rewrite, terminology
augmentation, first retrieval, candidate-answer expansion, second retrieval,
then MCQ generation. ``validate_flag=False`` and this module never invokes
``get_online_context``: online search and the LLM validator are both disabled.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OPTION_KEY = re.compile(r"option\s*(\d+)", re.IGNORECASE)
submit_prompt_flex: Any
Query: Any


def load_project_env() -> None:
    """Load local credentials without overriding an explicitly exported key."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)


def load_paper_modules() -> None:
    """Delay heavy paper imports so ``--help`` works before bootstrapping Python."""
    global Query, submit_prompt_flex
    from src.LLMs.LLM import submit_prompt_flex as paper_submit_prompt_flex
    from src.query import Query as PaperQuery

    submit_prompt_flex = paper_submit_prompt_flex
    Query = PaperQuery


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def option_lines(record: dict[str, Any]) -> list[str]:
    choices: list[tuple[int, str, Any]] = []
    for key, value in record.items():
        match = OPTION_KEY.fullmatch(str(key).strip())
        if match:
            choices.append((int(match.group(1)), str(key), value))
    if choices:
        return [f"{key}: {value}" for _, key, value in sorted(choices)]

    options = record.get("options")
    if isinstance(options, dict):
        return [f"{key}: {value}" for key, value in options.items()]
    if isinstance(options, list):
        return [str(value) for value in options]
    raise ValueError("No TeleQnA option fields found")


def option_label(value: str) -> str | None:
    """Normalize either ``Option 2`` or a bare ``2`` into ``Option 2``."""
    match = OPTION_KEY.search(value)
    if match:
        return f"Option {match.group(1)}"
    if value.strip().isdigit():
        return f"Option {value.strip()}"
    return None


def answer_option_only(question: Any, record: dict[str, Any], model: str) -> tuple[bool, str | None, str]:
    """Use the paper's final MCQ context but require a one-token-style answer."""
    context = "\n".join(question.context)
    options = "\n".join(option_lines(record))
    prompt = f"""
Select the correct answer to this multiple-choice question.

Question:
{question.query}

Retrieved context:
{context}

Original question:
{question.question}

Options:
{options}

Return exactly `Option <number>` and nothing else.
"""
    prediction = option_label(submit_prompt_flex(prompt, model=model, max_tokens=8).strip())
    ground_truth = option_label(str(record["answer"]))
    if ground_truth is None:
        raise ValueError(f"Cannot parse ground-truth option: {record['answer']!r}")
    return prediction == ground_truth, prediction, prompt


def run_paper_offline(record: dict[str, Any], model: str) -> dict[str, Any]:
    """Execute the published offline sequence without changing its retrieval logic."""
    question_text = str(record["question"])
    question = Query(question_text, [])
    concise_prompt = f"Rephrase the question to be clear and concise:\n\n{question.question}"
    question.query = submit_prompt_flex(concise_prompt, model=model).rstrip('"')
    question.def_TA_question()

    # This is the paper's local two-pass 3GPP path. It never requests online data.
    question.get_3GPP_context(k=10, model_name=model, validate_flag=False, UI_flag=False)
    is_correct, predicted, prompt = answer_option_only(question, record, model)
    return {
        "correct": is_correct,
        "predicted": predicted,
        "retrieval_sources": question.context_source,
        "retrieval_count": len(question.context),
        "answer_prompt": prompt,
    }


def existing_checkpoint(output: Path) -> tuple[set[int], int, int]:
    if not output.exists():
        return set(), 0, 0
    indices: set[int] = set()
    correct = 0
    attempted = 0
    with output.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if isinstance(row.get("correct"), bool):
                    indices.add(int(row["index"]))
                    attempted += 1
                    correct += row["correct"]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"Invalid JSONL checkpoint at {output}:{line_number}") from error
    return indices, correct, attempted


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "teleqna_3gpp_release.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "paper_offline_option_only_gpt-4o-mini_teleqna.jsonl",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent questions (1-4; default: 2).")
    parser.add_argument("--limit", type=int, help="Run only the first N uncompleted records.")
    parser.add_argument("--overwrite", action="store_true", help="Discard this JSONL checkpoint before starting.")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    load_project_env()
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in Telco-RAG_api/.env; no request has been sent.")
    if not args.dataset.is_file():
        parser.error(f"Dataset not found: {args.dataset}. Run prepare_paper_assets.py first.")
    corpus = PROJECT_ROOT / "3GPP-Release18"
    if not (corpus / "Documents").is_dir() or not (corpus / "Embeddings").is_dir():
        parser.error("Paper 3GPP corpus/embeddings not found. Run prepare_paper_assets.py first.")
    os.environ["TELCO_RAG_QUIET"] = "1"
    load_paper_modules()
    logging.getLogger().setLevel(logging.WARNING)

    with args.dataset.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        parser.error("Filtered TeleQnA dataset must be a JSON list.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output.exists():
        args.output.unlink()
    done, correct, attempted = existing_checkpoint(args.output)
    candidates = [(index, record) for index, record in enumerate(records) if index not in done]
    if args.limit is not None:
        candidates = candidates[: args.limit]

    run_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "dataset_sha256": file_sha256(args.dataset),
        "records_in_dataset": len(records),
        "model": args.model,
        "workers": args.workers,
        "paper_flow": {
            "online_search": False,
            "validator": False,
            "retrieval_k": 10,
            "retrieval_passes": 2,
            "embedding_model": "text-embedding-3-large",
            "embedding_dimensions": 1024,
            "final_answer_format": "Option <number> only",
        },
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2)
        handle.write("\n")

    started = time.monotonic()
    def evaluate(index: int, record: dict[str, Any]) -> dict[str, Any]:
        question_started = time.monotonic()
        row: dict[str, Any] = {
            "index": index,
            "question": record.get("question"),
            "answer": record.get("answer"),
            "model": args.model,
            "offline": True,
            "validator": False,
        }
        try:
            row.update(run_paper_offline(record, args.model))
        except Exception as error:  # preserve the checkpoint and continue the costly run
            row.update({"correct": None, "error": f"{type(error).__name__}: {error}"})
        row["duration_seconds"] = round(time.monotonic() - question_started, 3)

        return row

    executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="teleqna")
    try:
        futures = {executor.submit(evaluate, index, record): index for index, record in candidates}
        for future in as_completed(futures):
            index = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = {
                    "index": index,
                    "question": records[index].get("question"),
                    "answer": records[index].get("answer"),
                    "model": args.model,
                    "offline": True,
                    "validator": False,
                    "correct": None,
                    "error": f"WorkerError: {type(error).__name__}: {error}",
                }
            if isinstance(row.get("correct"), bool):
                attempted += 1
                correct += row["correct"]
            append_jsonl(args.output, row)
            accuracy = correct / attempted if attempted else 0.0
            prediction = row.get("predicted") or "ERROR"
            ground_truth = option_label(str(row.get("answer", ""))) or "UNKNOWN"
            print(
                f"{index + 1}/{len(records)}: predict: {prediction}, ground truth: {ground_truth}, "
                f"accuracy: {correct}/{attempted} ({accuracy:.2%})"
                + (f", error: {row['error']}" if row.get("error") else "")
            )
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        print("Interrupted: completed rows are checkpointed; rerun the same command to resume.")
        return 130
    else:
        executor.shutdown(wait=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
