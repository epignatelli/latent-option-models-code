#!/bin/bash -l
# SGE job: download + batch-convert nld-nao on Myriad, rsync each batch to
# a remote destination, then delete local npz files to stay within quota.
#
# Usage: qsub scripts/myriad_nao_convert.sh
#
# Configure the variables in the CONFIGURATION section before submitting.
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
#$ -l h_rt=72:0:0
#$ -l mem=16G
#$ -pe smp 48
#$ -l tmpfs=50G
#$ -N nao_convert
#$ -o logs/nao_convert.out
#$ -e logs/nao_convert.err
#$ -cwd

# --------------------------------------------------------------------------- #
# CONFIGURATION — edit before submitting
# --------------------------------------------------------------------------- #
DEST="uceeepi@<remote-hostname>:/scratch/uceeepi/lom/datasets/nle/nao"
WORKERS=48
BATCH_SIZE=5000        # players per batch; tune to keep output < 200 GB/batch
OUTPUT_DIR="$HOME/lom/datasets"
CODE_DIR="$HOME/repos/latent-option-models-code"

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
module load python3
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate lom

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

    # Rsync npz files (not the index — the destination builds its own).
    rsync -avz --progress \
        --exclude="index.npz" \
        "$NPZ_DIR/" \
        "$DEST/"

    echo "[$(date)] Rsync done. Deleting local npz files to free quota..."
    find "$NPZ_DIR" -maxdepth 1 -name "*.npz" ! -name "index.npz" -delete

    echo "[$(date)] Freed $(du -sh "$OUTPUT_DIR" | cut -f1) total used after cleanup."
done

# --------------------------------------------------------------------------- #
# Step 3: rsync the final index
# --------------------------------------------------------------------------- #
echo "[$(date)] Sending final index..."
rsync -avz "$OUTPUT_DIR/nle/nao/index.npz" "$DEST/index.npz"

echo "[$(date)] All done."
