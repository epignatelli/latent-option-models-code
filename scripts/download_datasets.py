"""Download NLD datasets for LOM pre-training.

Three datasets are supported:

  nld-aa    — NetHack Learning Dataset (Autoascend AI), 16 zip archives (~100 GB)
              https://dl.fbaipublicfiles.com/nld/nld-aa/
  nld-nao   — NetHack Learning Dataset (NetHack.alt.org), 41 zip archives (~500 GB)
              https://dl.fbaipublicfiles.com/nld/nld-nao/
  nao-top10 — NAO Top-10 processed dataset from DeepMind, single tar (~12 GB)
              https://storage.googleapis.com/dm_nethack/nao_top10.tar

NLD-AA and NLD-NAO are in NLE ttyrec format; an SQLite database is built after
extraction using the NLE dataset API.

NAO-TOP10 is pre-processed .npz tensors (no NLE database needed).

Usage:
    python -m scripts.download_datasets nld-aa    --output_dir /scratch/uceeepi/lom/datasets
    python -m scripts.download_datasets nld-nao   --output_dir /scratch/uceeepi/lom/datasets
    python -m scripts.download_datasets nao-top10 --output_dir /scratch/uceeepi/lom/datasets
"""

from __future__ import annotations

import argparse
import os
import tarfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


# --------------------------------------------------------------------------- #
# --- Dataset manifests ------------------------------------------------------ #
# --------------------------------------------------------------------------- #

_NLD_AA_BASE  = "https://dl.fbaipublicfiles.com/nld/nld-aa/"
_NLD_NAO_BASE = "https://dl.fbaipublicfiles.com/nld/nld-nao/"
_NAO_TOP10_URL = "https://storage.googleapis.com/dm_nethack/nao_top10.tar"


def _nld_aa_zips() -> list[str]:
    # nld-aa-dir-aa.zip … nld-aa-dir-ap.zip  (16 archives)
    return [f"nld-aa-dir-a{c}.zip" for c in "abcdefghijklmnop"]


def _nld_nao_zips() -> list[str]:
    # nld-nao-dir-aa.zip … nld-nao-dir-az.zip (26) + ba … bn (14) = 40 data
    # + nld-nao_xlogfiles.zip = 41 total
    suffixes = [f"a{c}" for c in "abcdefghijklmnopqrstuvwxyz"]
    suffixes += [f"b{c}" for c in "abcdefghijklmn"]
    zips = [f"nld-nao-dir-{s}.zip" for s in suffixes]
    zips.append("nld-nao_xlogfiles.zip")
    return zips


# --------------------------------------------------------------------------- #
# --- Shared helpers --------------------------------------------------------- #
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


def _untar(tar_path: str, dest_dir: str) -> None:
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(dest_dir)


def _populate_nle_db(unzipped_dir: str, db_path: str, dataset_name: str, use_altorg: bool) -> None:
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

    if use_altorg:
        nld.add_altorg_directory(unzipped_dir, dataset_name, filename=db_path)
    else:
        nld.add_nledata_directory(unzipped_dir, dataset_name, filename=db_path)


def _parallel_download(base_url: str, filenames: list[str], zip_dir: str, workers: int) -> None:
    print(f"Downloading {len(filenames)} archives to {zip_dir} ({workers} workers) ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download, base_url + name, os.path.join(zip_dir, name)): name
            for name in filenames
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


def _extract_zips(filenames: list[str], zip_dir: str, unzip_dir: str) -> None:
    print(f"\nExtracting {len(filenames)} archives to {unzip_dir} ...")
    for name in tqdm(filenames, unit="file"):
        _unzip(os.path.join(zip_dir, name), unzip_dir)


def _remove_zips(filenames: list[str], zip_dir: str) -> None:
    for name in filenames:
        path = os.path.join(zip_dir, name)
        if os.path.exists(path):
            os.remove(path)
    try:
        os.rmdir(zip_dir)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# --- Dataset-specific routines --------------------------------------------- #
# --------------------------------------------------------------------------- #

def _download_nld_aa(args: argparse.Namespace) -> None:
    filenames = _nld_aa_zips()
    zip_dir   = os.path.join(args.output_dir, "zips", "nld-aa")
    unzip_dir = os.path.join(args.output_dir, "nld-aa")
    db_path   = os.path.join(args.output_dir, "nld-aa.db")

    os.makedirs(zip_dir, exist_ok=True)
    os.makedirs(unzip_dir, exist_ok=True)

    _parallel_download(_NLD_AA_BASE, filenames, zip_dir, args.workers)
    _extract_zips(filenames, zip_dir, unzip_dir)

    if not args.keep_zips:
        _remove_zips(filenames, zip_dir)

    print(f"\nBuilding NLE database at {db_path} ...")
    _populate_nle_db(unzip_dir, db_path, "nld-aa", use_altorg=False)

    print(f"\nDone. Set data.nle_data_dir: {args.output_dir} in your experiment config.")


def _download_nld_nao(args: argparse.Namespace) -> None:
    filenames = _nld_nao_zips()
    zip_dir   = os.path.join(args.output_dir, "zips", "nld-nao")
    unzip_dir = os.path.join(args.output_dir, "nld-nao")
    db_path   = os.path.join(args.output_dir, "nld-nao.db")

    os.makedirs(zip_dir, exist_ok=True)
    os.makedirs(unzip_dir, exist_ok=True)

    _parallel_download(_NLD_NAO_BASE, filenames, zip_dir, args.workers)
    _extract_zips(filenames, zip_dir, unzip_dir)

    if not args.keep_zips:
        _remove_zips(filenames, zip_dir)

    print(f"\nBuilding NLE database at {db_path} ...")
    _populate_nle_db(unzip_dir, db_path, "nld-nao", use_altorg=True)

    print(f"\nDone. Set data.nle_data_dir: {args.output_dir} in your experiment config.")


def _download_nao_top10(args: argparse.Namespace) -> None:
    tar_dir    = os.path.join(args.output_dir, "zips", "nao-top10")
    tar_path   = os.path.join(tar_dir, "nao_top10.tar")
    extract_dir = os.path.join(args.output_dir, "nao-top10")

    os.makedirs(tar_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    print(f"Downloading nao_top10.tar (~11.8 GB) to {tar_path} ...")
    _download(_NAO_TOP10_URL, tar_path)

    print(f"\nExtracting to {extract_dir} ...")
    _untar(tar_path, extract_dir)

    if not args.keep_zips:
        os.remove(tar_path)
        try:
            os.rmdir(tar_dir)
        except OSError:
            pass

    print(f"\nDone. Set data.nle_data_dir: {args.output_dir} in your experiment config.")


# --------------------------------------------------------------------------- #
# --- Main ------------------------------------------------------------------ #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NLD datasets for LOM pre-training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="dataset", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output_dir", default="nle_data",
                        help="Root directory for the dataset")
    common.add_argument("--workers", type=int, default=4,
                        help="Parallel download workers (zip datasets only)")
    common.add_argument("--keep_zips", action="store_true",
                        help="Keep archives after extraction")

    sub.add_parser("nld-aa",    parents=[common],
                   help="NLD-AA (Autoascend AI gameplay, 16 zips)")
    sub.add_parser("nld-nao",   parents=[common],
                   help="NLD-NAO (NetHack.alt.org, 41 zips)")
    sub.add_parser("nao-top10", parents=[common],
                   help="NAO Top-10 processed .npz dataset from DeepMind (1 tar)")

    args = parser.parse_args()

    dispatch = {
        "nld-aa":    _download_nld_aa,
        "nld-nao":   _download_nld_nao,
        "nao-top10": _download_nao_top10,
    }
    dispatch[args.dataset](args)


if __name__ == "__main__":
    main()
