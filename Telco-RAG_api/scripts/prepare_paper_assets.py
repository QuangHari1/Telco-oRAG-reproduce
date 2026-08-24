#!/usr/bin/env python3
"""Download and validate the exact paper corpus plus the TeleQnA 3GPP subset.

This script deliberately downloads the published vectors instead of re-embedding
the corpus. The paper router and FAISS retrieval require these 1,024-dimension
``text-embedding-3-large`` vectors in their original series layout.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TELEQNA_REPO = "netop/TeleQnA"
TELEQNA_REVISION = "0eba715a43f0ab7e4d9d7e09ceb258642a149391"
CORPUS_REPO = "netop/3GPP-R18"
CORPUS_REVISION = "b8d598e50cada8aaa4de641abbec77bef6b51839"
RELEASE_TAG = re.compile(r"\[3GPP Release (1[4-9])\]")
EXPECTED_3GPP_QUESTIONS = 1810

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
CORPUS_DIR = PROJECT_ROOT / "3GPP-Release18"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def npy_shape(path: Path) -> tuple[int, ...]:
    """Read a NumPy header without importing NumPy into the asset downloader."""
    with path.open("rb") as handle:
        if handle.read(6) != bytes([0x93]) + b"NUMPY":
            raise RuntimeError(f"Not a NumPy array: {path}")
        major, _minor = handle.read(2)
        header_size = struct.unpack("<H" if major == 1 else "<I", handle.read(2 if major == 1 else 4))[0]
        header = ast.literal_eval(handle.read(header_size).decode("latin1"))
    return tuple(header["shape"])


def hf_download(repo: str, revision: str, destination: Path, include: str | None = None) -> None:
    command = [
        "hf",
        "download",
        repo,
        "--type",
        "dataset",
        "--revision",
        revision,
        "--local-dir",
        str(destination),
        "--max-workers",
        "4",
    ]
    if include:
        command.extend(["--include", include])
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return list(payload.values())
    raise TypeError(f"Unexpected TeleQnA JSON root: {type(payload).__name__}")


def prepare_teleqna() -> dict[str, Any]:
    source_dir = DATASETS_DIR / "teleqna_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    hf_download(TELEQNA_REPO, TELEQNA_REVISION, source_dir, "TeleQnA.json")

    source_path = source_dir / "TeleQnA.json"
    records = load_records(source_path)
    filtered = [record for record in records if RELEASE_TAG.search(str(record.get("question", "")))]
    if len(filtered) != EXPECTED_3GPP_QUESTIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_3GPP_QUESTIONS} tagged 3GPP Release questions at "
            f"{TELEQNA_REVISION}, found {len(filtered)}. Refusing an unpinned comparison."
        )

    output = DATASETS_DIR / "teleqna_3gpp_release.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(filtered, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    releases = sorted({match.group(1) for row in filtered for match in [RELEASE_TAG.search(row["question"])] if match})
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"repo": TELEQNA_REPO, "revision": TELEQNA_REVISION, "file": "TeleQnA.json"},
        "filter": RELEASE_TAG.pattern,
        "records": len(filtered),
        "releases": releases,
        "output": output.name,
        "sha256": sha256(output),
    }
    with (DATASETS_DIR / "teleqna_3gpp_release.manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def prepare_corpus() -> dict[str, Any]:
    hf_download(CORPUS_REPO, CORPUS_REVISION, CORPUS_DIR)
    documents = sorted((CORPUS_DIR / "Documents").glob("*.docx"))
    embeddings = sorted((CORPUS_DIR / "Embeddings").glob("Embeddings*.npy"))
    required = [CORPUS_DIR / "Documents.db", CORPUS_DIR / "Embeddings" / "EmbeddingsSummaries.npy"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing or len(documents) != 553 or len(embeddings) != 19:
        raise RuntimeError(
            f"Incomplete paper corpus: documents={len(documents)}, embeddings={len(embeddings)}, missing={missing}"
        )
    shapes = {path.stem.removeprefix("Embeddings"): npy_shape(path) for path in embeddings}
    invalid = {name: shape for name, shape in shapes.items() if shape != (0,) and shape[-1] != 1024}
    if invalid:
        raise RuntimeError(f"Unexpected paper embedding dimensions: {invalid}")
    return {
        "source": {"repo": CORPUS_REPO, "revision": CORPUS_REVISION},
        "documents": len(documents),
        "embedding_files": len(embeddings),
        "embedding_model": "text-embedding-3-large",
        "embedding_dimensions": 1024,
        "empty_embedding_series": sorted(name for name, shape in shapes.items() if shape == (0,)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-corpus", action="store_true", help="Only download/filter TeleQnA.")
    parser.add_argument("--skip-teleqna", action="store_true", help="Only download/validate the paper corpus.")
    args = parser.parse_args()
    if args.skip_corpus and args.skip_teleqna:
        parser.error("At least one asset must be selected.")

    manifest_path = DATASETS_DIR / "paper_reproduction_assets.manifest.json"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest: dict[str, Any] = json.load(handle)
    else:
        manifest = {}
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    if not args.skip_teleqna:
        manifest["teleqna"] = prepare_teleqna()
    if not args.skip_corpus:
        manifest["corpus"] = prepare_corpus()
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
