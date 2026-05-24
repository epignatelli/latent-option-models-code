#!/bin/bash -l
# SGE job: download + batch-convert all NLE datasets on Myriad, rsync each
# batch to a remote destination, then delete local npz files to stay within
# quota.
#
# Usage: qsub scripts/myriad_nao_convert.sh
#
# Prerequisites (run once on a Myriad login node before submitting):
#   git clone https://github.com/epignatelli/latent-option-models-code ~/repos/latent-option-models-code
#   mkdir -p ~/repos/latent-option-models-code/logs
#
# How it works:
#   For each dataset (nao-top10, nld-aa, nld-nao):
#   1. Download + extract raw data once (skipped on restart).
#   2. Loop: convert BATCH_SIZE groups, rsync npz files to DEST_BASE,
#      delete local npz files (keep index).
#   3. Retry any OOM failures at 10 workers.
#   4. Rsync the final index.
#
# Peak disk usage: largest raw dataset (~500 GB nld-nao) + one batch (~20 GB).

# --------------------------------------------------------------------------- #
# SGE directives
# --------------------------------------------------------------------------- #
#$ -S /bin/bash
#$ -l h_rt=48:0:0
#$ -l mem=16G
#$ -pe smp 36
#$ -ac allow=B
#$ -l tmpfs=50G
#$ -N nle_convert
#$ -o logs/nle_convert.out
#$ -e logs/nle_convert.err
#$ -cwd

# --------------------------------------------------------------------------- #
# CONFIGURATION — edit before submitting
# --------------------------------------------------------------------------- #
DEST_BASE="uceeepi@bologna.ee.ucl.ac.uk:/scratch/uceeepi/lom/datasets"
WORKERS=36
BATCH_SIZE=5000
OUTPUT_DIR="$HOME/lom/datasets"
RAW_DIR="/dev/shm/lom_$$"
CODE_DIR="$HOME/repos/latent-option-models-code"

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
module load python/miniconda3/24.3.0-0
source "$UCL_CONDA_PATH/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^lom-convert "; then
    conda create -n lom-convert -c conda-forge -y \
        python=3.11 \
        cmake make autoconf libtool pkg-config flex bison bzip2 zlib \
        "numpy>=1.26,<2" "tqdm>=4.66" "psutil>=5.9" pip
    conda run -n lom-convert pip install "tyro>=0.8" "nle>=0.9"
fi
conda activate lom-convert

mkdir -p "$OUTPUT_DIR" logs
if mkdir -p "$RAW_DIR" && touch "$RAW_DIR/.write_test" 2>/dev/null; then
    rm -f "$RAW_DIR/.write_test"
    OUTPUT_DIR="$RAW_DIR"
    echo "OUTPUT_DIR=$OUTPUT_DIR (local /dev/shm)"
else
    echo "WARNING: cannot write to $RAW_DIR — falling back to $OUTPUT_DIR (NFS)"
fi

# --------------------------------------------------------------------------- #
# run_dataset <dataset> <npz_subdir> <dest> [--skip-db]
#
#   dataset     — name passed to prepare_data.py (nld-nao, nld-aa, nao-top10)
#   npz_subdir  — relative path under OUTPUT_DIR where npz files are written
#   dest        — rsync destination (host:path)
#   --skip-db   — pass for datasets that have a db stage (nld-nao, nld-aa)
# --------------------------------------------------------------------------- #
run_dataset() {
    local dataset=$1
    local npz_subdir=$2
    local dest=$3
    local skip_db_flag=${4:-}

    local npz_dir="$OUTPUT_DIR/$npz_subdir"
    mkdir -p "$npz_dir"

    # Step 1: download + extract (+ db if applicable)
    echo "[$(date)] [$dataset] Downloading and extracting..."
    python "$CODE_DIR/scripts/prepare_data.py" "$dataset" \
        --output-dir "$OUTPUT_DIR" \
        --workers "$WORKERS" \
        --skip-convert \
        --skip-index

    # Step 2: batch convert → rsync → delete
    local iteration=0
    while true; do
        iteration=$((iteration + 1))
        echo "[$(date)] [$dataset] Batch $iteration (max $BATCH_SIZE groups)..."

        python "$CODE_DIR/scripts/prepare_data.py" "$dataset" \
            --output-dir "$OUTPUT_DIR" \
            --workers "$WORKERS" \
            --max-groups "$BATCH_SIZE" \
            --skip-download \
            --skip-extract \
            $skip_db_flag

        local n_new
        n_new=$(find "$npz_dir" -maxdepth 1 -name "*.npz" ! -name "index.npz" | wc -l)
        if [ "$n_new" -eq 0 ]; then
            echo "[$(date)] [$dataset] No new files — all groups converted."
            break
        fi

        echo "[$(date)] [$dataset] $n_new files converted. Rsyncing..."
        rsync -avz --progress --exclude="index.npz" "$npz_dir/" "$dest/"
        find "$npz_dir" -maxdepth 1 -name "*.npz" ! -name "index.npz" -delete
        echo "[$(date)] [$dataset] Rsynced and deleted. Disk used: $(du -sh "$OUTPUT_DIR" | cut -f1)"
    done

    # Step 3: retry OOM failures at 10 workers
    local errors_file="$npz_dir/errors.txt"
    if [ -s "$errors_file" ]; then
        local n_before
        n_before=$(wc -l < "$errors_file")
        echo "[$(date)] [$dataset] Retrying $n_before failures at 10 workers..."
        python "$CODE_DIR/scripts/prepare_data.py" "$dataset" \
            --output-dir "$OUTPUT_DIR" \
            --workers 10 \
            --skip-download \
            --skip-extract \
            $skip_db_flag

        local n_new
        n_new=$(find "$npz_dir" -maxdepth 1 -name "*.npz" ! -name "index.npz" | wc -l)
        local n_after
        n_after=$([ -s "$errors_file" ] && wc -l < "$errors_file" || echo 0)

        if [ "$n_new" -eq 0 ]; then
            echo "[$(date)] [$dataset] Retry made no progress ($n_after errors remain)."
        else
            rsync -avz --progress --exclude="index.npz" "$npz_dir/" "$dest/"
            find "$npz_dir" -maxdepth 1 -name "*.npz" ! -name "index.npz" -delete
            echo "[$(date)] [$dataset] Retry done. $n_after error entries in log."
        fi
    else
        echo "[$(date)] [$dataset] No errors to retry."
    fi

    # Step 4: rsync final index
    if [ -f "$npz_dir/index.npz" ]; then
        echo "[$(date)] [$dataset] Sending final index..."
        rsync -avz "$npz_dir/index.npz" "$dest/index.npz"
    else
        echo "[$(date)] [$dataset] No index.npz found — skipping."
    fi

    # Step 5: free /dev/shm before next dataset
    echo "[$(date)] [$dataset] Cleaning up RAW_DIR..."
    rm -rf "${RAW_DIR:?}/${dataset}" "${RAW_DIR:?}/zips/${dataset}" "${RAW_DIR:?}/${dataset}.db"

    echo "[$(date)] [$dataset] Done."
}

# --------------------------------------------------------------------------- #
# Run all datasets
# --------------------------------------------------------------------------- #
run_dataset "nao-top10" "nle/nao-top10" "$DEST_BASE/nle/nao-top10"
run_dataset "nld-aa"    "nle/aa"        "$DEST_BASE/nle/aa"        "--skip-db"
run_dataset "nld-nao"   "nle/nao"       "$DEST_BASE/nle/nao"       "--skip-db"

echo "[$(date)] All datasets done."
