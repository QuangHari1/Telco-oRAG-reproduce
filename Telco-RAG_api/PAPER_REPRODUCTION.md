# Offline paper reproduction

This setup evaluates the repository's published offline path, not the web UI
or online-search pipeline. It preserves the paper implementation's sequence:

1. rewrite query and expand 3GPP terminology;
2. route to 3GPP series, retrieve 10 local chunks;
3. generate candidate answers and perform the second local retrieval;
4. have the answer model return the MCQ option.

`validate_flag=False` is the value in the upstream
`Telco-RAG_paper_version/pipeline_offline.py`; neither online search nor the
LLM context validator runs. The router's published, precomputed
`text-embedding-3-large` vectors (1,024 dimensions) are downloaded rather
than regenerated.

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

The runner defaults to `gpt-4o-mini`, two workers, and an `Option N`-only
final-answer prompt. It writes one JSON object per completed question to
`results/paper_offline_option_only_gpt-4o-mini_teleqna.jsonl`; this JSONL file
is the checkpoint, so re-run the same command to resume after a crash. The
terminal prints `index/1810: predict, ground truth, accuracy` as each worker
finishes. Completed right/wrong answers are skipped on resume; error rows are
kept in the JSONL audit trail and retried. Use `--workers 4` only if RAM and
OpenAI rate limits permit it. For a small paid smoke test, use `--limit 3`; use `--overwrite` only when
intentionally discarding that result checkpoint.

This is pipeline-faithful, but not a numerically exact reproduction of every
paper table: it uses the requested 1,810 tagged TeleQnA subset and a
non-pinned hosted GPT-4o mini model. The paper's final TeleQnA evaluation set
was 1,840 questions.
