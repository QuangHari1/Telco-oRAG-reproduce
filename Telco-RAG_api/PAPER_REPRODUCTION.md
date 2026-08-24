# Offline reproduction of the paper baseline

This is a runnable, modified offline baseline: it evaluates the repository's
published local-3GPP path, not the web UI or the online-search pipeline. Its
retrieval/control flow follows the paper implementation:

1. rewrite query and expand 3GPP terminology;
2. route to 3GPP series, retrieve 10 local chunks;
3. generate candidate answers and perform the second local retrieval;
4. answer the MCQ from the final local context.

`validate_flag=False` is the value in the upstream
`Telco-RAG_paper_version/pipeline_offline.py`; neither online search nor the
LLM context validator runs. The router's published, precomputed
`text-embedding-3-large` vectors (1,024 dimensions) are downloaded rather
than regenerated.

The runner intentionally makes the final MCQ response machine-readable:
`Option <number>` only. This is safer to score than the upstream
`check_question()` prompt, but means the final-generation prompt is not
byte-for-byte identical to the repository reference.

## Prepared data

`scripts/prepare_paper_assets.py` pins both Hugging Face revisions:

- `netop/3GPP-R18@b8d598e50cada8aaa4de641abbec77bef6b51839`: 553 DOCX,
  `Documents.db`, and 19 embedding arrays.
- `netop/TeleQnA@0eba715a43f0ab7e4d9d7e09ceb258642a149391`, filtered only by
  `[3GPP Release 14]` through `[3GPP Release 19]`: 1,810 records.

All downloaded data and results are ignored by Git. The asset and dataset
manifests record the revisions and SHA-256 values.

## Run

```bash
cd Telco-RAG_api
UV_CACHE_DIR=/tmp/uv-cache uv sync
cp .env.example .env
# Edit .env: OPENAI_API_KEY=...
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/prepare_paper_assets.py
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/run_offline_teleqna.py --workers 2
```

`uv sync` installs the `hf` CLI used by the asset script. If either public Hub
dataset becomes gated, authenticate once with `hf auth login` before preparing
assets.

The runner defaults to `gpt-4o-mini`, two workers, and an `Option N`-only
final-answer prompt. It records every attempt in
`results/paper_offline_option_only_gpt-4o-mini_teleqna.jsonl`; successful
indices are the checkpoint, so re-run the same command to resume after a
crash. Failed rows stay in the audit trail and are retried. The terminal prints
`index/1810: predict, ground truth, accuracy` as each worker finishes. Use
`--workers 4` only if RAM and OpenAI rate limits permit it. For a small paid
smoke test, use `--limit 3`; use `--overwrite` only when intentionally
discarding that result checkpoint.

## What this result means

Do not compare its accuracy directly with a paper table. The paper uses 2,000
synthetic Release-18 questions for optimization and a curated 1,840-question
3GPP Standard evaluation set; this setup uses the requested 1,810-question
tag-filtered TeleQnA subset (Releases 14--19). It also uses a non-pinned hosted
`gpt-4o-mini` model, whereas the paper's default experiments use GPT-3.5 unless
otherwise stated. Record the output manifest, dataset SHA-256, model, and date
whenever reporting a run.
