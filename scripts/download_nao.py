"""Download and set up the NLD-NAO dataset for LOM pre-training.

Based on: https://github.com/NetHack-LE/nle/blob/main/DATASET.md

Downloads 41 zip archives (40 data + xlogfiles) from FAIR's AWS S3,
unzips them into a local directory, and populates an NLE SQLite database.
Zip files are removed after successful extraction unless --keep_zips is set.

Usage:
    python -m scripts.download_nao --output_dir /scratch/uceeepi/lom/datasets
    python -m scripts.download_nao --output_dir ./data --workers 8 --keep_zips
"""

from __future__ import annotations

import argparse
import os
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


# --------------------------------------------------------------------------- #
# --- Dataset manifest ------------------------------------------------------ #
# --------------------------------------------------------------------------- #

_BASE_URL = "https://dl.fbaipublicfiles.com/nld/nld-nao/"

# 40 data archives: aa–az (26) + ba–bn (14)
def _data_suffixes() -> list[str]:
    suffixes = [f"a{c}" for c in "abcdefghijklmnopqrstuvwxyz"]   # aa–az
    suffixes += [f"b{c}" for c in "abcdefghijklmn"]               # ba–bn
    return suffixes

DATA_ZIPS = [f"nld-nao-dir-{s}.zip" for s in _data_suffixes()]
XLOG_ZIP  = "nld-nao_xlogfiles.zip"
ALL_ZIPS  = DATA_ZIPS + [XLOG_ZIP]


# --------------------------------------------------------------------------- #
# --- Helpers --------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def _download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    tmp = dest + ".tmp"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.rename(tmp, dest)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _unzip(zip_path: str, dest_dir: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def _populate_db(unzipped_dir: str, db_path: str, dataset_name: str) -> None:
    try:
        import nle.dataset as nld
        import nle.dataset.db as nld_db
    except ImportError:
        raise ImportError(
            "NLE is not installed.\n"
            "  pip install git+https://github.com/NetHack-LE/nle.git@main"
        )

    if not nld_db.exists(filename=db_path):
        nld_db.create(filename=db_path)

    nld.add_altorg_directory(unzipped_dir, dataset_name, filename=db_path)


# --------------------------------------------------------------------------- #
# --- Main ------------------------------------------------------------------ #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and set up the NLD-NAO dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir", default="nle_data",
        help="Root directory for the dataset (nao.db will be created here)",
    )
    parser.add_argument(
        "--dataset_name", default="nao",
        help="Name to register in the NLE database",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel download workers",
    )
    parser.add_argument(
        "--keep_zips", action="store_true",
        help="Keep zip archives after extraction",
    )
    args = parser.parse_args()

    zip_dir     = os.path.join(args.output_dir, "zips")
    unzip_dir   = os.path.join(args.output_dir, "nld-nao")
    db_path     = os.path.join(args.output_dir, "nao.db")

    os.makedirs(zip_dir, exist_ok=True)
    os.makedirs(unzip_dir, exist_ok=True)

    # --- Download -----------------------------------------------------------
    print(f"Downloading {len(ALL_ZIPS)} archives to {zip_dir} ({args.workers} workers) ...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download, _BASE_URL + name, os.path.join(zip_dir, name)): name
            for name in ALL_ZIPS
        }
        with tqdm(total=len(futures), unit="file") as bar:
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed to download {name}: {exc}") from exc
                bar.set_postfix(file=name)
                bar.update(1)

    # --- Unzip --------------------------------------------------------------
    print(f"\nExtracting {len(ALL_ZIPS)} archives to {unzip_dir} ...")
    for name in tqdm(ALL_ZIPS, unit="file"):
        _unzip(os.path.join(zip_dir, name), unzip_dir)

    # --- Clean up zips ------------------------------------------------------
    if not args.keep_zips:
        for name in ALL_ZIPS:
            path = os.path.join(zip_dir, name)
            if os.path.exists(path):
                os.remove(path)
        try:
            os.rmdir(zip_dir)
        except OSError:
            pass  # non-empty if --keep_zips was toggled mid-run

    # --- Populate DB --------------------------------------------------------
    print(f"\nBuilding NLE database at {db_path} ...")
    _populate_db(unzip_dir, db_path, args.dataset_name)

    print(f"\nDone. Dataset ready at {args.output_dir}")
    print(f"Set data.nle_data_dir: {args.output_dir} in your experiment config.")


if __name__ == "__main__":
    main()
