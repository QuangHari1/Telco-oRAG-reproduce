#!/usr/bin/env python3
"""Run the last TeleQnA records with the upstream offline MCQ prompt.

This runner intentionally preserves the local retrieval path in
``Telco-RAG_paper_version/pipeline_offline.py`` while replacing only the final
strict ``Option <number>`` prompt used by ``run_offline_teleqna.py``.  Its
prompt and list-based option scorer match ``src.generate.check_question`` and
the released TeleQnA experiment.  No ``max_tokens`` cap is applied to the
final answer, as in the reference implementation.
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

OPTION_PATTERN = re.compile(r"option\D*(\d+)")
Query: Any
submit_prompt_flex: Any


def load_project_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)


def load_paper_modules() -> None:
    global Query, submit_prompt_flex
    from src.LLMs.LLM import submit_prompt_flex as paper_submit_prompt_flex
    from src.query import Query as PaperQuery

    Query = PaperQuery
    submit_prompt_flex = paper_submit_prompt_flex


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upstream_option_lines(record: dict[str, Any]) -> list[str]:
    """Match the list construction in the released TeleQnA experiment."""
    return [f"{key}: {value}" for key, value in record.items() if "option" in str(key)]


def find_option_numbers(text: str) -> list[str]:
    """Match ``src.generate.find_option_number`` exactly."""
    return OPTION_PATTERN.findall(text.lower())


def answer_with_reference_prompt(question: Any, record: dict[str, Any], model: str) -> dict[str, Any]:
    """Use the upstream final MCQ prompt and its original list equality scorer."""
    options_text = "\n".join(upstream_option_lines(record))
    content = "\n".join(question.context)
    prompt = f"""
        Please provide the answers to the following multiple choice question.
        {question.query}
        
        Considering the following context:
        {content}
        
        Please provide the answers to the following multiple choice question.
        {question.question}
        
        Options:
        Write only the option number corresponding to the correct answer:\n{options_text}
        
        Answer format should be: Answer option <option_id>
        """
    raw_output = submit_prompt_flex(prompt, model=model)
    normalized_output = raw_output.replace('"\n', '",\n')
    predicted_numbers = find_option_numbers(normalized_output)
    expected_numbers = find_option_numbers(str(record["answer"]))
    if not expected_numbers:
        raise ValueError(f"Cannot parse expected option: {record['answer']!r}")
    return {
        "correct": predicted_numbers == expected_numbers,
        "raw_output": raw_output,
        "normalized_output": normalized_output,
        "predicted_option_numbers": predicted_numbers,
        "expected_option_numbers": expected_numbers,
        "answer_prompt": prompt,
    }


def run_reference_offline(record: dict[str, Any], model: str) -> dict[str, Any]:
    question = Query(str(record["question"]), [])
    concise_prompt = f"Rephrase the question to be clear and concise:\n\n{question.question}"
    question.query = submit_prompt_flex(concise_prompt, model=model).rstrip('"')
    question.def_TA_question()
    question.get_3GPP_context(k=10, model_name=model, validate_flag=False, UI_flag=False)
    result = answer_with_reference_prompt(question, record, model)
    result.update(
        {
            "retrieval_sources": question.context_source,
            "retrieval_count": len(question.context),
        }
    )
    return result


def completed_indices(path: Path) -> tuple[set[int], int, int]:
    if not path.exists():
        return set(), 0, 0
    completed: set[int] = set()
    correct = 0
    attempted = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if isinstance(row.get("correct"), bool):
                    completed.add(int(row["index"]))
                    attempted += 1
                    correct += row["correct"]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"Invalid JSONL checkpoint at {path}:{line_number}") from error
    return completed, correct, attempted


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets" / "teleqna_3gpp_release.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "paper_offline_reference_prompt_gpt-4o-mini_last200.jsonl",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent questions (1-4; default: 2).")
    parser.add_argument("--last", type=int, default=200, help="Evaluate the final N records (default: 200).")
    parser.add_argument("--overwrite", action="store_true", help="Discard this runner's separate checkpoint.")
    args = parser.parse_args()
    if args.last <= 0:
        parser.error("--last must be positive")
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

    with args.dataset.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        parser.error("Filtered TeleQnA dataset must be a JSON list.")
    if args.last > len(records):
        parser.error(f"--last cannot exceed dataset size ({len(records)}).")

    os.environ["TELCO_RAG_QUIET"] = "1"
    load_paper_modules()
    logging.getLogger().setLevel(logging.WARNING)
    start_index = len(records) - args.last
    selected_indices = range(start_index, len(records))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output.exists():
        args.output.unlink()
    done, correct, attempted = completed_indices(args.output)
    candidates = [(index, records[index]) for index in selected_indices if index not in done]

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "dataset_sha256": file_sha256(args.dataset),
        "records_in_dataset": len(records),
        "selected_indices": {"start": start_index, "end": len(records) - 1, "count": args.last},
        "model": args.model,
        "workers": args.workers,
        "paper_flow": {
            "online_search": False,
            "validator": False,
            "retrieval_k": 10,
            "retrieval_passes": 2,
            "embedding_model": "text-embedding-3-large",
            "embedding_dimensions": 1024,
            "final_answer_prompt": "upstream src.generate.check_question",
            "final_answer_max_tokens": None,
            "scorer": "upstream find_option_number list equality",
        },
    }
    with args.output.with_suffix(args.output.suffix + ".manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    def evaluate(index: int, record: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        row: dict[str, Any] = {
            "index": index,
            "question": record.get("question"),
            "answer": record.get("answer"),
            "model": args.model,
            "offline": True,
            "validator": False,
            "prompt_mode": "upstream_reference",
        }
        try:
            row.update(run_reference_offline(record, args.model))
        except Exception as error:
            row.update({"correct": None, "error": f"{type(error).__name__}: {error}"})
        row["duration_seconds"] = round(time.monotonic() - started, 3)
        return row

    executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="teleqna-reference")
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
                    "prompt_mode": "upstream_reference",
                    "correct": None,
                    "error": f"WorkerError: {type(error).__name__}: {error}",
                }
            if isinstance(row.get("correct"), bool):
                attempted += 1
                correct += row["correct"]
            append_jsonl(args.output, row)
            prediction = row.get("predicted_option_numbers", [])
            expected = row.get("expected_option_numbers", [])
            accuracy = correct / attempted if attempted else 0.0
            print(
                f"{index + 1}/{len(records)}: predict: {prediction or 'ERROR'}, "
                f"ground truth: {expected or 'UNKNOWN'}, accuracy: {correct}/{attempted} ({accuracy:.2%})"
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
