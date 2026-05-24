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
#$ -j y
#$ -cwd

# --------------------------------------------------------------------------- #
# CONFIGURATION — edit before submitting
# --------------------------------------------------------------------------- #
DEST_BASE="uceeepi@bologna.ee.ucl.ac.uk:/scratch/uceeepi/lom/datasets"
WORKERS=36
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

# Background monitor — logs CPU/RAM/net every 60s to logs/monitor.log
(while true; do
    echo "=== $(date) ==="
    free -h
    top -b -n 1 -u uceeepi | head -20
    awk 'NR>2 {printf "%-10s rx:%s tx:%s\n",$1,$2,$10}' /proc/net/dev
    echo ""
    sleep 60
done) >> logs/monitor.log 2>&1 &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT

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

    # Step 1: download, extract, and convert in one shot
    echo "[$(date)] [$dataset] Starting..."
    python "$CODE_DIR/scripts/prepare_data.py" "$dataset" \
        --output-dir "$OUTPUT_DIR" \
        --workers "$WORKERS" \
        $skip_db_flag

    # Step 3: retry OOM failures at 10 workers
    local errors_file="$npz_dir/errors.txt"
    if [ -s "$errors_file" ]; then
        echo "[$(date)] [$dataset] Retrying $(wc -l < "$errors_file") failures at 10 workers..."
        python "$CODE_DIR/scripts/prepare_data.py" "$dataset" \
            --output-dir "$OUTPUT_DIR" \
            --workers 10 \
            --skip-download \
            --skip-extract \
            $skip_db_flag
    else
        echo "[$(date)] [$dataset] No errors to retry."
    fi

    # Step 4: rsync everything to bologna
    echo "[$(date)] [$dataset] Rsyncing to $dest..."
    rsync -avz --progress "$npz_dir/" "$dest/"

    # Step 5: free /dev/shm before next dataset
    echo "[$(date)] [$dataset] Cleaning up..."
    rm -rf "${RAW_DIR:?}/${dataset}" "${RAW_DIR:?}/zips/${dataset}" "${RAW_DIR:?}/${dataset}.db"

    echo "[$(date)] [$dataset] Done."
}

# --------------------------------------------------------------------------- #
# Run all datasets
# --------------------------------------------------------------------------- #
# run_dataset "nao-top10" "nle/nao-top10" "$DEST_BASE/nle/nao-top10"
# run_dataset "nld-aa"    "nle/aa"        "$DEST_BASE/nle/aa"        "--skip-db"
run_dataset "nld-nao"   "nle/nao"       "$DEST_BASE/nle/nao"       "--skip-db"

echo "[$(date)] All datasets done."
