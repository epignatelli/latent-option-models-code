#!/bin/bash -l
# SGE job: download + batch-convert nld-nao on Myriad, rsync each batch to
# a remote destination, then delete local npz files to stay within quota.
#
# Usage: qsub scripts/myriad_nao_convert.sh
#
# Configure the variables in the CONFIGURATION section before submitting.
#
# Prerequisites (run once on a Myriad login node before submitting):
#   git clone https://github.com/epignatelli/latent-option-models-code ~/repos/latent-option-models-code
#   mkdir -p ~/logs
#
# How it works:
#   1. Download + extract raw nld-nao data once (~500 GB).
#   2. Loop: convert BATCH_SIZE players, rsync their npz files to DEST,
#      delete the local npz files (but keep the index).
#   3. Because prepare_data.py skips players already in the index, each
#      batch processes only new players — no duplicate work across iterations.
#   4. Repeat until all players are done.
#
# Peak disk usage: raw data (~500 GB) + one batch of npz (~10-20 GB) < 1 TB.

# --------------------------------------------------------------------------- #
# SGE directives
# --------------------------------------------------------------------------- #
#$ -S /bin/bash
#$ -l h_rt=48:0:0
#$ -l mem=16G
#$ -pe smp 32
#$ -l tmpfs=50G
#$ -N nao_convert
#$ -o logs/nao_convert.out
#$ -e logs/nao_convert.err
#$ -cwd

# --------------------------------------------------------------------------- #
# CONFIGURATION — edit before submitting
# --------------------------------------------------------------------------- #
DEST="uceeepi@bologna.ee.ucl.ac.uk:/scratch/uceeepi/lom/datasets/nle/nao"
WORKERS=32
BATCH_SIZE=5000        # players per batch; tune to keep output < 200 GB/batch
OUTPUT_DIR="$HOME/lom/datasets"
CODE_DIR="$HOME/repos/latent-option-models-code"

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
module load python/3.9.6-gnu-10.2.0
module load cmake/3.21.1  # required to build nle

# One-time dependency install (safe to re-run; pip skips already-installed).
pip install --user --quiet \
    nle numpy tqdm psutil tyro wandb

mkdir -p "$OUTPUT_DIR" logs

# --------------------------------------------------------------------------- #
# Step 1: download + extract raw nld-nao (run once; skipped on restart)
# --------------------------------------------------------------------------- #
echo "[$(date)] Downloading nld-nao raw data..."
python "$CODE_DIR/scripts/prepare_data.py" nld-nao \
    --output-dir "$OUTPUT_DIR" \
    --skip-convert \
    --skip-index

# --------------------------------------------------------------------------- #
# Step 2: batch convert → rsync → delete, until no players remain
# --------------------------------------------------------------------------- #
ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    echo "[$(date)] === Batch $ITERATION (max $BATCH_SIZE players) ==="

    # Convert a batch; index is updated in-place by prepare_data.py.
    python "$CODE_DIR/scripts/prepare_data.py" nld-nao \
        --output-dir "$OUTPUT_DIR" \
        --workers "$WORKERS" \
        --max-groups "$BATCH_SIZE" \
        --skip-download \
        --skip-extract \
        --skip-db

    NPZ_DIR="$OUTPUT_DIR/nle/nao"

    # Count newly written npz files (anything that is NOT the index).
    N_NEW=$(find "$NPZ_DIR" -maxdepth 1 -name "*.npz" ! -name "index.npz" | wc -l)
    if [ "$N_NEW" -eq 0 ]; then
        echo "[$(date)] No new files — all players converted. Done."
        break
    fi
    echo "[$(date)] Converted $N_NEW player files. Rsyncing to $DEST..."

    rsync -avz --progress \
        --exclude="index.npz" \
        "$NPZ_DIR/" \
        "$DEST/"

    echo "[$(date)] Rsync done. Deleting local npz files to free quota..."
    find "$NPZ_DIR" -maxdepth 1 -name "*.npz" ! -name "index.npz" -delete

    echo "[$(date)] Freed $(du -sh "$OUTPUT_DIR" | cut -f1) total used after cleanup."
done

# --------------------------------------------------------------------------- #
# Step 3: retry pass at 4 workers for any OOM-failed players
# --------------------------------------------------------------------------- #
ERRORS_FILE="$OUTPUT_DIR/nle/nao/errors.txt"
if [ -s "$ERRORS_FILE" ]; then
    N_ERRORS_BEFORE=$(wc -l < "$ERRORS_FILE")
    echo "[$(date)] Retrying $N_ERRORS_BEFORE failed players at 10 workers..."
    python "$CODE_DIR/scripts/prepare_data.py" nld-nao \
        --output-dir "$OUTPUT_DIR" \
        --workers 10 \
        --skip-download \
        --skip-extract \
        --skip-db

    NPZ_DIR="$OUTPUT_DIR/nle/nao"
    N_NEW=$(find "$NPZ_DIR" -maxdepth 1 -name "*.npz" ! -name "index.npz" | wc -l)
    N_ERRORS_AFTER=$([ -s "$ERRORS_FILE" ] && wc -l < "$ERRORS_FILE" || echo 0)

    if [ "$N_NEW" -eq 0 ] || [ "$N_ERRORS_AFTER" -eq "$N_ERRORS_BEFORE" ]; then
        echo "[$(date)] Retry made no progress ($N_ERRORS_AFTER errors remain) — skipping."
    else
        rsync -avz --progress --exclude="index.npz" "$NPZ_DIR/" "$DEST/"
        find "$NPZ_DIR" -maxdepth 1 -name "*.npz" ! -name "index.npz" -delete
        echo "[$(date)] Retry done. $N_ERRORS_AFTER players still failed."
    fi
else
    echo "[$(date)] No errors to retry."
fi

# --------------------------------------------------------------------------- #
# Step 4: rsync the final index
# --------------------------------------------------------------------------- #
echo "[$(date)] Sending final index..."
rsync -avz "$OUTPUT_DIR/nle/nao/index.npz" "$DEST/index.npz"

echo "[$(date)] All done."
