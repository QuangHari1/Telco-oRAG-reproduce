#!/usr/bin/env python3
"""Create paired charts and a PDF comparing paper-offline and newbaseline runs.

The paper JSONL embeds question text and uses ``correct``.  Newbaseline JSONL
uses ``question_id`` and ``is_correct``; the baseline manifest identifies the
TeleQnA source used to resolve its question IDs. Only shared completed questions
are compared, so the same script works for a 200-question slice or a full run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import textwrap
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/telco-rag-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_RESULTS = PROJECT_ROOT / "results" / "paper_offline_option_only_gpt-4o-mini_teleqna.jsonl"
DEFAULT_BASELINE_RESULTS = Path("/home/quanghari/project/vht/Telco-RAG/newbaseline/results/teleqna/paper-baseline-gsma-rel18.jsonl")
RELEASE_PATTERN = re.compile(r"\[3GPP Release (\d+)\]", re.IGNORECASE)
PAPER_COLOR, BASELINE_COLOR, MUTED_COLOR, GRID_COLOR = "#2563EB", "#EA580C", "#64748B", "#CBD5E1"
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of", "on", "or", "the",
    "this", "that", "to", "what", "when", "which", "with", "according", "following", "does", "do", "under", "using", "used",
    "can", "feature", "features", "happens", "maximum", "responsible", "use",
    "release", "3gpp", "specification", "specifications", "standard", "standards", "teleqna",
}


@dataclass(frozen=True)
class RunRecord:
    question: str
    question_id: str | None
    category: str
    release: str
    correct: bool
    prediction: str | None
    expected: str | None
    source_index: int | None


@dataclass(frozen=True)
class PairedRecord:
    question: str
    question_id: str | None
    category: str
    release: str
    paper_correct: bool
    baseline_correct: bool
    paper_prediction: str | None
    baseline_prediction: str | None
    expected: str | None
    paper_index: int | None


def normalize_question(value: str) -> str:
    return " ".join(value.split())


def release_for(question: str) -> str:
    match = RELEASE_PATTERN.search(question)
    return match.group(1) if match else "Unknown"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Result file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON at {path}:{number}") from error
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected object at {path}:{number}")
            rows.append(row)
    return rows


def load_question_source(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return {str(key): value for key, value in payload.items() if isinstance(value, dict)}
    if isinstance(payload, list):
        return {f"question {index}": value for index, value in enumerate(payload) if isinstance(value, dict)}
    raise RuntimeError(f"Unsupported dataset root: {type(payload).__name__}")


def resolve_dataset_path(results_path: Path, explicit: Path | None, manifest: Path | None) -> Path:
    if explicit is not None:
        return explicit
    manifest_path = manifest or results_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError("Supply --baseline-dataset or keep a manifest next to --baseline-results.")
    with manifest_path.open(encoding="utf-8") as handle:
        dataset_path = json.load(handle).get("dataset_path")
    if not isinstance(dataset_path, str):
        raise RuntimeError(f"Manifest has no dataset_path: {manifest_path}")
    return Path(dataset_path)


def latest_paper(
    rows: list[dict[str, Any]], source: dict[str, dict[str, Any]]
) -> tuple[dict[str, RunRecord], list[str]]:
    source_rows = list(source.items())
    question_ids_by_text: dict[str, list[str]] = defaultdict(list)
    for question_id, detail in source_rows:
        question = detail.get("question")
        if isinstance(question, str):
            question_ids_by_text[normalize_question(question)].append(question_id)
    records: dict[str, RunRecord] = {}
    unresolved: list[str] = []
    for row in rows:
        question = row.get("question")
        if not isinstance(question, str) or not isinstance(row.get("correct"), bool):
            continue
        source_index = row.get("index") if isinstance(row.get("index"), int) else None
        question_id: str | None = None
        detail: dict[str, Any] | None = None
        if source_index is not None and 0 <= source_index < len(source_rows):
            candidate_id, candidate = source_rows[source_index]
            candidate_question = candidate.get("question")
            if isinstance(candidate_question, str) and normalize_question(candidate_question) == normalize_question(question):
                question_id, detail = candidate_id, candidate
        if question_id is None:
            candidates = question_ids_by_text.get(normalize_question(question), [])
            if len(candidates) == 1:
                question_id, detail = candidates[0], source[candidates[0]]
            else:
                unresolved.append(str(source_index))
                continue
        records[question_id] = RunRecord(
            question=question,
            question_id=question_id,
            category=str(detail.get("category", "Unknown")),
            release=release_for(question),
            correct=row["correct"],
            prediction=row.get("predicted") if isinstance(row.get("predicted"), str) else None,
            expected=row.get("answer") if isinstance(row.get("answer"), str) else None,
            source_index=source_index,
        )
    return records, unresolved


def completed_baseline(
    rows: list[dict[str, Any]], source: dict[str, dict[str, Any]]
) -> tuple[dict[str, RunRecord], list[str]]:
    records: dict[str, RunRecord] = {}
    unresolved: list[str] = []
    for row in rows:
        question_id = row.get("question_id")
        if not isinstance(row.get("is_correct"), bool):
            continue
        if not isinstance(question_id, str) or question_id not in source:
            unresolved.append(str(question_id))
            continue
        detail = source[question_id]
        question = detail.get("question")
        if not isinstance(question, str):
            unresolved.append(question_id)
            continue
        records[question_id] = RunRecord(
            question=question,
            question_id=question_id,
            category=str(detail.get("category", "Unknown")),
            release=release_for(question),
            correct=row["is_correct"],
            prediction=row.get("predicted_option") if isinstance(row.get("predicted_option"), str) else None,
            expected=row.get("expected_option") if isinstance(row.get("expected_option"), str) else None,
            source_index=None,
        )
    return records, unresolved


def write_csv(path: Path, rows: list[PairedRecord]) -> None:
    fields = list(PairedRecord.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_breakdown(path: Path, groups: dict[str, list[PairedRecord]], dimension: str) -> None:
    fields = ["dimension", "group", "questions", "paper_accuracy", "baseline_accuracy", "accuracy_delta_baseline_minus_paper", "both_correct", "paper_only", "baseline_only", "both_wrong"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
            total = len(rows)
            paper_correct = sum(row.paper_correct for row in rows)
            baseline_correct = sum(row.baseline_correct for row in rows)
            writer.writerow({
                "dimension": dimension,
                "group": label,
                "questions": total,
                "paper_accuracy": round(paper_correct / total, 4),
                "baseline_accuracy": round(baseline_correct / total, 4),
                "accuracy_delta_baseline_minus_paper": round((baseline_correct - paper_correct) / total, 4),
                "both_correct": sum(row.paper_correct and row.baseline_correct for row in rows),
                "paper_only": sum(row.paper_correct and not row.baseline_correct for row in rows),
                "baseline_only": sum(not row.paper_correct and row.baseline_correct for row in rows),
                "both_wrong": sum(not row.paper_correct and not row.baseline_correct for row in rows),
            })


def finish(figure: plt.Figure, png: Path, pdf: PdfPages) -> None:
    figure.tight_layout()
    figure.savefig(png, dpi=180, bbox_inches="tight")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def overview(summary: dict[str, Any], png: Path, pdf: PdfPages) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 7.5))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    axis.set_title("Paired benchmark overview", fontsize=18, fontweight="bold", pad=18)
    paper, baseline, paired = summary["paper"], summary["baseline"], summary["paired_correctness"]
    axis.text(0.03, 0.92, f"Matched completed questions: {summary['matched_questions']}", fontsize=13, weight="bold")
    axis.text(0.03, 0.86, f"{paper['label']}: {paper['correct']}/{paper['total']} ({paper['accuracy']:.1%})", color=PAPER_COLOR, fontsize=12)
    axis.text(0.03, 0.81, f"{baseline['label']}: {baseline['correct']}/{baseline['total']} ({baseline['accuracy']:.1%})", color=BASELINE_COLOR, fontsize=12)
    axis.add_patch(Circle((0.42, 0.47), 0.22, color=PAPER_COLOR, alpha=0.35))
    axis.add_patch(Circle((0.58, 0.47), 0.22, color=BASELINE_COLOR, alpha=0.35))
    for x, key in ((0.30, "paper_only"), (0.50, "both_correct"), (0.70, "baseline_only")):
        axis.text(x, 0.47, str(paired[key]), ha="center", va="center", fontsize=16, weight="bold")
    axis.text(0.33, 0.20, f"{paper['label']} only correct", ha="center", fontsize=10, color=PAPER_COLOR)
    axis.text(0.50, 0.15, "Both correct", ha="center", fontsize=10)
    axis.text(0.67, 0.20, f"{baseline['label']} only correct", ha="center", fontsize=10, color=BASELINE_COLOR)
    axis.text(0.50, 0.72, f"Both wrong: {paired['both_wrong']}", ha="center", fontsize=13, weight="bold", color=MUTED_COLOR)
    axis.text(0.03, 0.04, "Circle areas are illustrative; labels are exact counts of correct-answer sets.", fontsize=9, color=MUTED_COLOR)
    finish(figure, png, pdf)


def paired_matrix(summary: dict[str, Any], png: Path, pdf: PdfPages) -> None:
    paired = summary["paired_correctness"]
    values = [[paired["both_correct"], paired["paper_only"]], [paired["baseline_only"], paired["both_wrong"]]]
    figure, axis = plt.subplots(figsize=(8, 6.5))
    image = axis.imshow(values, cmap="Blues")
    figure.colorbar(image, ax=axis, shrink=0.8, label="Question count")
    axis.set(xticks=[0, 1], xticklabels=["Baseline correct", "Baseline wrong"], yticks=[0, 1], yticklabels=["Paper correct", "Paper wrong"])
    axis.set_title("Paired correctness matrix", fontsize=16, fontweight="bold")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(values[row][column]), ha="center", va="center", fontsize=20, weight="bold")
    finish(figure, png, pdf)


def grouped_accuracy(groups: dict[str, list[PairedRecord]], title: str, png: Path, pdf: PdfPages) -> None:
    labels = sorted(groups, key=lambda label: (-len(groups[label]), label))
    paper = [sum(row.paper_correct for row in groups[label]) / len(groups[label]) for label in labels]
    baseline = [sum(row.baseline_correct for row in groups[label]) / len(groups[label]) for label in labels]
    positions = list(range(len(labels)))
    figure, axis = plt.subplots(figsize=(10.5, max(4.5, len(labels) * 0.75 + 2)))
    width = 0.36
    axis.barh([position + width / 2 for position in positions], paper, height=width, color=PAPER_COLOR, label="Paper offline")
    axis.barh([position - width / 2 for position in positions], baseline, height=width, color=BASELINE_COLOR, label="New baseline")
    axis.set(yticks=positions, yticklabels=[f"{label} (n={len(groups[label])})" for label in labels], xlim=(0, 1), xlabel="Accuracy", title=title)
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right")
    finish(figure, png, pdf)


def outcomes_by_category(groups: dict[str, list[PairedRecord]], png: Path, pdf: PdfPages) -> None:
    labels = sorted(groups, key=lambda label: (-len(groups[label]), label))
    parts = {
        "Both correct": [sum(row.paper_correct and row.baseline_correct for row in groups[label]) for label in labels],
        "Paper only": [sum(row.paper_correct and not row.baseline_correct for row in groups[label]) for label in labels],
        "Baseline only": [sum(not row.paper_correct and row.baseline_correct for row in groups[label]) for label in labels],
        "Both wrong": [sum(not row.paper_correct and not row.baseline_correct for row in groups[label]) for label in labels],
    }
    figure, axis = plt.subplots(figsize=(10.5, max(4.5, len(labels) * 0.75 + 2)))
    left = [0] * len(labels)
    for (name, values), color in zip(parts.items(), ["#22C55E", PAPER_COLOR, BASELINE_COLOR, "#94A3B8"]):
        axis.barh(labels, values, left=left, label=name, color=color)
        left = [prior + value for prior, value in zip(left, values)]
    axis.set(title="Where each run succeeds or fails, by category", xlabel="Question count")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right")
    finish(figure, png, pdf)


def error_prone_terms(rows: list[PairedRecord], png: Path, pdf: PdfPages) -> None:
    """Compare recurring literal question terms with enough support in the slice."""
    term_rows: dict[str, list[PairedRecord]] = defaultdict(list)
    for row in rows:
        terms = {
            term.lower()
            for term in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", row.question)
            if term.lower() not in STOPWORDS and not term.isdigit()
        }
        for term in terms:
            term_rows[term].append(row)
    candidates = [
        (term, values)
        for term, values in term_rows.items()
        if len(values) >= 5 and sum(not row.baseline_correct for row in values) >= 2
        and sum(not row.baseline_correct for row in values) > sum(not row.paper_correct for row in values)
    ]
    candidates.sort(
        key=lambda item: (
            (sum(not row.baseline_correct for row in item[1]) - sum(not row.paper_correct for row in item[1])) / len(item[1]),
            len(item[1]),
            item[0],
        ),
        reverse=True,
    )
    selected = candidates[:12]
    figure, axis = plt.subplots(figsize=(10.5, max(4.5, len(selected) * 0.55 + 2)))
    if not selected:
        axis.axis("off")
        axis.text(0.5, 0.5, "No recurring term has enough observations for this slice.", ha="center", va="center", fontsize=13)
    else:
        labels = [f"{term} (n={len(values)})" for term, values in selected]
        paper_error = [sum(not row.paper_correct for row in values) / len(values) for _, values in selected]
        baseline_error = [sum(not row.baseline_correct for row in values) / len(values) for _, values in selected]
        positions = list(range(len(selected)))
        width = 0.36
        axis.barh([position + width / 2 for position in positions], paper_error, height=width, color=PAPER_COLOR, label="Paper offline")
        axis.barh([position - width / 2 for position in positions], baseline_error, height=width, color=BASELINE_COLOR, label="New baseline")
        axis.set(yticks=positions, yticklabels=labels, xlim=(0, 1), xlabel="Error rate", title="Terms where the new baseline has a higher error rate")
        axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.legend(loc="lower right")
        axis.text(0, -0.15, "Exploratory grouping by literal question terms; each term occurs in at least 5 questions. Small n is not a causal finding.", transform=axis.transAxes, fontsize=8.5, color=MUTED_COLOR)
    finish(figure, png, pdf)


def example_pages(pdf: PdfPages, title: str, rows: list[PairedRecord]) -> None:
    for start in range(0, len(rows), 8):
        figure, axis = plt.subplots(figsize=(11.7, 8.3))
        axis.axis("off")
        page = rows[start : start + 8]
        axis.set_title(f"{title} ({start + 1}–{start + len(page)} of {len(rows)})", fontsize=15, fontweight="bold", pad=16)
        y = 0.93
        for row in page:
            text = (
                f"{row.question_id or 'unknown id'} | Rel-{row.release} | {row.category}\n"
                f"Q: {textwrap.shorten(row.question, width=138, placeholder=' …')}\n"
                f"Expected: {row.expected or 'unknown'} | paper: {row.paper_prediction or 'unparsed'} | baseline: {row.baseline_prediction or 'unparsed'}"
            )
            axis.text(0.02, y, text, transform=axis.transAxes, va="top", fontsize=8.6, wrap=True)
            y -= 0.115
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-results", type=Path, default=DEFAULT_PAPER_RESULTS)
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--baseline-dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "paper_vs_paper_baseline_gsma_rel18_analysis")
    parser.add_argument("--paper-label", default="Paper offline")
    parser.add_argument("--baseline-label", default="Paper baseline GSMA Rel-18")
    args = parser.parse_args()

    paper_rows, baseline_rows = load_jsonl(args.paper_results), load_jsonl(args.baseline_results)
    dataset_path = resolve_dataset_path(args.baseline_results, args.baseline_dataset, args.baseline_manifest)
    source = load_question_source(dataset_path)
    paper, unresolved_paper = latest_paper(paper_rows, source)
    baseline, unresolved = completed_baseline(baseline_rows, source)
    common = sorted(set(paper) & set(baseline), key=lambda key: (paper[key].source_index is None, paper[key].source_index, key))
    if not common:
        raise RuntimeError("No completed questions matched between the two inputs.")
    pairs = [
        PairedRecord(
            question=paper[key].question, question_id=baseline[key].question_id or paper[key].question_id,
            category=baseline[key].category, release=baseline[key].release, paper_correct=paper[key].correct,
            baseline_correct=baseline[key].correct, paper_prediction=paper[key].prediction,
            baseline_prediction=baseline[key].prediction, expected=baseline[key].expected or paper[key].expected,
            paper_index=paper[key].source_index,
        )
        for key in common
    ]
    paper_only = [row for row in pairs if row.paper_correct and not row.baseline_correct]
    baseline_only = [row for row in pairs if row.baseline_correct and not row.paper_correct]
    both_correct = [row for row in pairs if row.paper_correct and row.baseline_correct]
    both_wrong = [row for row in pairs if not row.paper_correct and not row.baseline_correct]
    by_category: dict[str, list[PairedRecord]] = defaultdict(list)
    by_release: dict[str, list[PairedRecord]] = defaultdict(list)
    for row in pairs:
        by_category[row.category].append(row)
        by_release[row.release].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("all_matched_questions", pairs), ("paper_only_correct", paper_only), ("baseline_only_correct", baseline_only), ("both_wrong", both_wrong)):
        write_csv(args.output_dir / f"{name}.csv", rows)
    write_breakdown(args.output_dir / "accuracy_by_category.csv", by_category, "category")
    write_breakdown(args.output_dir / "accuracy_by_release.csv", by_release, "3GPP release")
    summary = {
        "paper": {"label": args.paper_label, "input_rows": len(paper_rows), "unique_completed_questions": len(paper), "correct": sum(row.paper_correct for row in pairs), "total": len(pairs), "accuracy": sum(row.paper_correct for row in pairs) / len(pairs)},
        "baseline": {"label": args.baseline_label, "input_rows": len(baseline_rows), "unique_completed_questions": len(baseline), "correct": sum(row.baseline_correct for row in pairs), "total": len(pairs), "accuracy": sum(row.baseline_correct for row in pairs) / len(pairs)},
        "matched_questions": len(pairs), "paper_unmatched_completed": len(set(paper) - set(baseline)), "baseline_unmatched_completed": len(set(baseline) - set(paper)), "unresolved_paper_indices": unresolved_paper, "unresolved_baseline_question_ids": unresolved,
        "paired_correctness": {"both_correct": len(both_correct), "paper_only": len(paper_only), "baseline_only": len(baseline_only), "both_wrong": len(both_wrong)},
        "input_files": {"paper_results": str(args.paper_results), "baseline_results": str(args.baseline_results), "baseline_dataset": str(dataset_path)},
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    with PdfPages(args.output_dir / "report.pdf") as pdf:
        overview(summary, args.output_dir / "01_correct_set_overlap.png", pdf)
        paired_matrix(summary, args.output_dir / "02_paired_correctness.png", pdf)
        grouped_accuracy(by_category, "Accuracy by TeleQnA category", args.output_dir / "03_accuracy_by_category.png", pdf)
        grouped_accuracy(by_release, "Accuracy by 3GPP release", args.output_dir / "04_accuracy_by_release.png", pdf)
        outcomes_by_category(by_category, args.output_dir / "05_outcomes_by_category.png", pdf)
        error_prone_terms(pairs, args.output_dir / "06_error_prone_terms.png", pdf)
        example_pages(pdf, f"Questions {args.paper_label} gets right and {args.baseline_label} misses", paper_only)
        example_pages(pdf, f"Questions {args.baseline_label} gets right and {args.paper_label} misses", baseline_only)
        example_pages(pdf, "Questions both runs miss", both_wrong)
    print(json.dumps(summary, indent=2))
    print(f"Wrote report and charts to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
