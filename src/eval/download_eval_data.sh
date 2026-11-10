#!/usr/bin/env bash
# Download the evaluation datasets into <repo>/datasets/.
# Downloaded data is gitignored; see .gitignore.
#
# Usage: src/eval/download_eval_data.sh
# Safe to re-run: the Hugging Face CLI skips files that are already complete.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASETS_DIR="$REPO_ROOT/datasets"

# dbpedia-openai-1M-1536-angular:
# 1M DBpedia entities embedded with text-embedding-ada-002 (1536 dims, cosine).
DBPEDIA_REPO="KShivendu/dbpedia-entities-openai-1M"
DBPEDIA_DIR="$DATASETS_DIR/dbpedia-openai-1M"

# Use the Hugging Face CLI if it's installed; otherwise run it through uvx.
if command -v hf >/dev/null 2>&1; then
    HF=(hf)
elif command -v uvx >/dev/null 2>&1; then
    HF=(uvx --from huggingface_hub hf)
else
    echo "error: need either the 'hf' CLI (pip install huggingface_hub) or 'uvx' (uv)." >&2
    exit 1
fi

mkdir -p "$DBPEDIA_DIR"
echo "Downloading $DBPEDIA_REPO -> $DBPEDIA_DIR (~9 GB, ~15 GB after preparing)"
"${HF[@]}" download "$DBPEDIA_REPO" --repo-type dataset --local-dir "$DBPEDIA_DIR"

# Convert to flat binary files + exact ground truth, used by `cargo run --release -- eval`.
# gt.ibin is written last, so if it exists the conversion already finished.
if [[ -f "$DBPEDIA_DIR/gt.ibin" ]]; then
    echo "Eval files already prepared, delete $DBPEDIA_DIR/gt.ibin to redo them."
else
    PREPARE="$REPO_ROOT/src/eval/prepare_dbpedia.py"
    if command -v uv >/dev/null 2>&1; then
        uv run --no-project --with numpy --with pyarrow python -I "$PREPARE" "$DBPEDIA_DIR"
    else
        # needs numpy and pyarrow installed
        python3 -I "$PREPARE" "$DBPEDIA_DIR"
    fi
fi

echo "Done."
du -sh "$DBPEDIA_DIR"
