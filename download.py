"""
Automatic download of the weights PatchET needs at inference time:
  - ESM-2 (esm2_t30_150M_UR50D) backbone files from the Hugging Face Hub -> esm150/
  - PatchET task checkpoints from Zenodo                               -> checkpoint/<task>/

The ESM-2 backbone is frozen during training, yet some released checkpoints also
contain its weights. Those `pretrain_model.*` tensors are stripped so every
checkpoint keeps only the PatchET weights; the backbone is always loaded from esm150/.

Files that are already present are never downloaded again. Can also be run
directly to pre-fetch everything, e.g. before going offline:

    python download.py --tasks opt stability range
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import tarfile
import tempfile
import urllib.request
import zipfile
from typing import Dict, List, Optional

from tqdm import tqdm


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ESM_DIR = os.path.join(REPO_DIR, "esm150")
DEFAULT_CHECKPOINT_DIR = os.path.join(REPO_DIR, "checkpoint")

ESM_REPO_ID = "facebook/esm2_t30_150M_UR50D"
ESM_FILES = ["config.json", "model.safetensors", "special_tokens_map.json",
             "tokenizer_config.json", "vocab.txt"]

# https://doi.org/10.5281/zenodo.23160814
ZENODO_RECORD_ID = "23160814"
ZENODO_API = "https://zenodo.org/api/records/{record_id}"

TASK_NAMES = ["opt", "stability", "range"]
BACKBONE_PREFIX = "pretrain_model"   # attribute holding the frozen ESM-2 in models/patchet*.py
CONFIG_NAME = "model_config.yaml"
WEIGHTS_NAME = "model.safetensors"
ARCHIVE_EXTS = (".zip", ".tar", ".tar.gz", ".tgz")


# ─────────────────────────────────────────────────────────────
# ESM-2 backbone
# ─────────────────────────────────────────────────────────────
def ensure_esm(esm_dir: str = DEFAULT_ESM_DIR, download: bool = True) -> str:
    """Make sure the ESM-2 backbone files exist in `esm_dir`, downloading them if needed."""
    missing = [f for f in ESM_FILES if not os.path.exists(os.path.join(esm_dir, f))]
    if not missing:
        return esm_dir
    if not download:
        raise FileNotFoundError(
            f"ESM-2 files missing from {esm_dir}: {', '.join(missing)}. "
            f"Download them from https://huggingface.co/{ESM_REPO_ID} or rerun without --no_download."
        )

    from huggingface_hub import snapshot_download

    print(f"Downloading ESM-2 backbone ({ESM_REPO_ID}) to {esm_dir} ...")
    snapshot_download(repo_id=ESM_REPO_ID, local_dir=esm_dir, allow_patterns=ESM_FILES)
    return esm_dir


# ─────────────────────────────────────────────────────────────
# PatchET checkpoints (Zenodo)
# ─────────────────────────────────────────────────────────────
def task_checkpoint_paths(task: str, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR) -> Dict[str, str]:
    task_dir = os.path.join(checkpoint_dir, task)
    return {"config": os.path.join(task_dir, CONFIG_NAME),
            "weights": os.path.join(task_dir, WEIGHTS_NAME)}


def _has_checkpoint(task: str, checkpoint_dir: str) -> bool:
    return all(os.path.exists(p) for p in task_checkpoint_paths(task, checkpoint_dir).values())


def ensure_checkpoints(tasks: List[str], checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
                       record_id: str = ZENODO_RECORD_ID, download: bool = True) -> None:
    """
    Make sure `checkpoint/<task>/{model_config.yaml, model.safetensors}` exist for every task,
    and that the weights hold only PatchET parameters (no frozen ESM-2 backbone).
    """
    _fetch_checkpoints(tasks, checkpoint_dir, record_id, download)
    for task in tasks:
        weights = task_checkpoint_paths(task, checkpoint_dir)["weights"]
        removed = strip_backbone_weights(weights)
        if removed:
            print(f"  [{task}] removed {removed} frozen ESM-2 tensors from {weights}")


def _fetch_checkpoints(tasks: List[str], checkpoint_dir: str, record_id: str, download: bool) -> None:
    missing = [t for t in tasks if not _has_checkpoint(t, checkpoint_dir)]
    if not missing:
        return
    if not download:
        raise FileNotFoundError(
            f"PatchET checkpoint(s) missing for task(s) {missing} in {checkpoint_dir}. "
            f"Download them from https://doi.org/10.5281/zenodo.{record_id} or rerun without --no_download."
        )

    print(f"Fetching PatchET checkpoint(s) for {missing} from Zenodo record {record_id} ...")
    files = _list_zenodo_files(record_id)

    # Case 1: the task's config and weights are uploaded as individual files
    # (Zenodo has no folders, so they are told apart by the task name in the file name).
    for task in list(missing):
        config_file = _pick(files, task, (".yaml", ".yml"))
        weights_file = _pick(files, task, (".safetensors",))
        if config_file and weights_file:
            paths = task_checkpoint_paths(task, checkpoint_dir)
            _download(config_file, paths["config"])
            _download(weights_file, paths["weights"])
            missing.remove(task)

    # Case 2: the checkpoints come as archive(s), e.g. opt.zip or a single checkpoint.zip.
    # Archives named after a missing task are tried first, then the remaining ones.
    archives = [f for f in files if f["key"].lower().endswith(ARCHIVE_EXTS)]
    archives.sort(key=lambda f: not any(_mentions(f["key"], t) for t in missing))
    for archive in archives:
        if not missing:
            break
        named_tasks = [t for t in TASK_NAMES if _mentions(archive["key"], t)]
        if named_tasks and not set(named_tasks) & set(missing):
            continue
        _install_from_archive(archive, checkpoint_dir)
        missing = [t for t in missing if not _has_checkpoint(t, checkpoint_dir)]

    if missing:
        raise RuntimeError(
            f"Could not find checkpoint(s) for task(s) {missing} in Zenodo record {record_id} "
            f"(files: {[f['key'] for f in files]}). Please download them manually from "
            f"https://doi.org/10.5281/zenodo.{record_id} into {checkpoint_dir}/<task>/."
        )


def _list_zenodo_files(record_id: str) -> List[dict]:
    """Return [{key, url, size, md5}] for every file in a Zenodo record."""
    url = ZENODO_API.format(record_id=record_id)
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        record = json.load(response)

    entries = record.get("files", [])
    if isinstance(entries, dict):   # InvenioRDM-style {"entries": [...]} or {"entries": {key: {...}}}
        entries = entries.get("entries", [])
        if isinstance(entries, dict):
            entries = list(entries.values())

    files = []
    for entry in entries:
        key = entry.get("key") or entry.get("filename")
        links = entry.get("links", {})
        file_url = links.get("content") or links.get("self") or links.get("download")
        if not file_url:
            file_url = f"https://zenodo.org/records/{record_id}/files/{key}?download=1"
        checksum = entry.get("checksum") or ""
        md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
        files.append({"key": key, "url": file_url, "size": entry.get("size") or entry.get("filesize"), "md5": md5})
    if not files:
        raise RuntimeError(f"Zenodo record {record_id} lists no files ({url}).")
    return files


def _mentions(name: str, task: str) -> bool:
    """True if `task` appears as a separate token in a file or folder name (e.g. 'opt_model.safetensors')."""
    return task in re.split(r"[^a-z0-9]+", name.lower())


def _pick(files: List[dict], task: str, exts) -> Optional[dict]:
    matches = [f for f in files if f["key"].lower().endswith(exts) and _mentions(f["key"], task)]
    return matches[0] if len(matches) == 1 else None


def _download(file: dict, dest: str) -> None:
    """Stream a file to `dest` (via a .part file) and verify its MD5 checksum when Zenodo provides one."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    md5 = hashlib.md5()
    with urllib.request.urlopen(file["url"], timeout=60) as response, open(tmp, "wb") as out, \
            tqdm(total=file.get("size"), unit="B", unit_scale=True, unit_divisor=1024,
                 desc=f"  {file['key']}", leave=False) as bar:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            md5.update(chunk)
            bar.update(len(chunk))

    if file.get("md5") and md5.hexdigest() != file["md5"]:
        os.remove(tmp)
        raise RuntimeError(f"Checksum mismatch for {file['key']}: expected {file['md5']}, got {md5.hexdigest()}.")
    os.replace(tmp, dest)
    print(f"  downloaded {file['key']} -> {dest}")


def _install_from_archive(archive: dict, checkpoint_dir: str) -> None:
    """Download and extract an archive, then copy every task checkpoint found in it into `checkpoint_dir`."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=checkpoint_dir, prefix=".download-") as tmp:
        archive_path = os.path.join(tmp, os.path.basename(archive["key"]))
        _download(archive, archive_path)

        extract_dir = os.path.join(tmp, "extracted")
        _safe_extract(archive_path, extract_dir)

        for root, _, filenames in os.walk(extract_dir):
            if CONFIG_NAME not in filenames or WEIGHTS_NAME not in filenames:
                continue
            rel_parts = os.path.relpath(root, extract_dir).split(os.sep)
            # The task is the innermost folder named after one, else the archive's own name.
            names = list(reversed(rel_parts)) + [archive["key"]]
            task = next((t for name in names for t in TASK_NAMES if _mentions(name, t)), None)
            if task is None or _has_checkpoint(task, checkpoint_dir):
                continue
            paths = task_checkpoint_paths(task, checkpoint_dir)
            os.makedirs(os.path.dirname(paths["config"]), exist_ok=True)
            shutil.copyfile(os.path.join(root, CONFIG_NAME), paths["config"])
            shutil.copyfile(os.path.join(root, WEIGHTS_NAME), paths["weights"])
            print(f"  installed '{task}' checkpoint -> {os.path.dirname(paths['config'])}")


def strip_backbone_weights(weights_path: str) -> int:
    """
    Remove the frozen ESM-2 tensors (`pretrain_model.*`) from a safetensors file in place,
    keeping only the PatchET weights. Returns the number of tensors removed (0 = unchanged).

    The file is rewritten at the byte level (header + raw tensor bytes), so this needs
    neither torch nor enough memory to hold the checkpoint, and preserves dtypes exactly.
    """
    try:
        with open(weights_path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
    except (struct.error, ValueError) as e:
        raise RuntimeError(f"{weights_path} is not a valid safetensors file: {e}") from e
    data_start = 8 + header_len

    metadata = header.pop("__metadata__", None)
    keep = {k: v for k, v in header.items() if k.split(".")[0] != BACKBONE_PREFIX}
    removed = len(header) - len(keep)
    if removed == 0:
        return 0
    if not keep:
        raise RuntimeError(f"{weights_path} contains only backbone weights; refusing to empty it.")

    # Pack the kept tensors contiguously, in their original order.
    new_header, offset = {}, 0
    order = sorted(keep, key=lambda k: keep[k]["data_offsets"][0])
    for key in order:
        begin, end = keep[key]["data_offsets"]
        new_header[key] = {**keep[key], "data_offsets": [offset, offset + end - begin]}
        offset += end - begin
    if metadata is not None:
        new_header["__metadata__"] = metadata

    header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)   # safetensors pads the header to 8 bytes

    tmp = weights_path + ".part"
    with open(weights_path, "rb") as src, open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for key in order:
            begin, end = keep[key]["data_offsets"]
            src.seek(data_start + begin)
            remaining = end - begin
            while remaining:
                chunk = src.read(min(remaining, 1 << 24))
                if not chunk:
                    raise RuntimeError(f"{weights_path} is truncated (tensor '{key}').")
                out.write(chunk)
                remaining -= len(chunk)
    os.replace(tmp, weights_path)
    return removed


def _safe_extract(archive_path: str, dest: str) -> None:
    """Extract a zip/tar archive, refusing members that would land outside `dest`."""
    dest = os.path.realpath(dest)

    def check(name: str) -> None:
        target = os.path.realpath(os.path.join(dest, name))
        if target != dest and not target.startswith(dest + os.sep):
            raise RuntimeError(f"Refusing to extract unsafe path from archive: {name}")

    if archive_path.lower().endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            for name in zf.namelist():
                check(name)
            zf.extractall(dest)
    else:
        with tarfile.open(archive_path) as tf:
            members = tf.getmembers()
            for member in members:
                check(member.name)
                if member.issym() or member.islnk() or member.isdev():
                    raise RuntimeError(f"Refusing to extract link/device from archive: {member.name}")
            tf.extractall(dest, members=members)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the ESM-2 backbone and PatchET checkpoints (stripped to PatchET weights only).")
    parser.add_argument("--tasks", type=str, nargs="+", choices=TASK_NAMES, default=TASK_NAMES,
                        help="Task checkpoints to download (default: all).")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help="Where task checkpoints are stored (default: checkpoint/).")
    parser.add_argument("--esm_dir", type=str, default=DEFAULT_ESM_DIR,
                        help="Where the ESM-2 backbone is stored (default: esm150/).")
    parser.add_argument("--zenodo_record", type=str, default=ZENODO_RECORD_ID,
                        help=f"Zenodo record holding the checkpoints (default: {ZENODO_RECORD_ID}).")
    args = parser.parse_args()

    ensure_esm(args.esm_dir)
    ensure_checkpoints(args.tasks, args.checkpoint_dir, args.zenodo_record)
    print("All weights are in place; task checkpoints hold PatchET weights only.")


if __name__ == "__main__":
    main()
