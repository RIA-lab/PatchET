"""
PatchET Inference Script
Predicts three thermal properties for protein sequences from a FASTA file:
  - Topt   : optimal temperature         (checkpoint/opt/)
  - Stability: thermostability threshold  (checkpoint/stability/)
  - Range  : [T_low, T_high] active range (checkpoint/range/)

Usage:
    python inference.py --fasta proteins.fasta --output predictions.csv
"""

import argparse
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import EsmTokenizer

from models import load_model
from utils import load_config, load_weight, read_fasta, replace_noncanonical


# ─────────────────────────────────────────────────────────────
# Task definitions
# ─────────────────────────────────────────────────────────────
TASKS = {
    "opt":       {"config": "checkpoint/opt/model_config.yaml",
                  "weights": "checkpoint/opt/model.safetensors",
                  "output_cols": ["topt"]},
    "stability": {"config": "checkpoint/stability/model_config.yaml",
                  "weights": "checkpoint/stability/model.safetensors",
                  "output_cols": ["t_stability"]},
    "range":     {"config": "checkpoint/range/model_config.yaml",
                  "weights": "checkpoint/range/model.safetensors",
                  "output_cols": ["t_low", "t_high"]},
}


# ─────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PatchET: predict Topt, stability, and temperature range from a FASTA file."
    )
    parser.add_argument(
        "--fasta", type=str, required=True,
        help="Path to the input FASTA file."
    )
    parser.add_argument(
        "--output", type=str, default="predictions.csv",
        help="Path to the output CSV file (default: predictions.csv)."
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+",
        choices=list(TASKS.keys()), default=["opt"],
        help="Which tasks to run (default: opt). Use multiple values for more tasks, e.g. --tasks opt stability range."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for inference (default: 16)."
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device (default: auto)."
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(choice)


def parse_uniprot_accession(header: str) -> str:
    """
    Extract the UniProt accession from a FASTA header line.

    Supports standard UniProt formats:
      sp|ACCESSION|ENTRY_NAME ...
      tr|ACCESSION|ENTRY_NAME ...

    Falls back to the first whitespace-delimited token if the header
    does not match the UniProt pipe-delimited format.
    """
    match = re.match(r"^(?:sp|tr)\|([A-Za-z0-9_-]+)\|", header)
    if match:
        return match.group(1)
    # Fallback: use everything before the first space
    return header.split()[0] if header.strip() else header


def normalize_sequence(value: object) -> Optional[str]:
    """Strip whitespace, upper-case, replace non-canonical AAs."""
    if not isinstance(value, str):
        return None
    seq = re.sub(r"\s+", "", value.strip().upper())
    if not seq:
        return None
    return replace_noncanonical(seq, replace_char="X")


def load_task_model(task_name: str, device: torch.device):
    """Instantiate the model for a task and load its checkpoint weights."""
    task_cfg = TASKS[task_name]
    config_path  = task_cfg["config"]
    weights_path = task_cfg["weights"]

    for path, label in [(config_path, "config"), (weights_path, "weights")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[{task_name}] {label} file not found: {os.path.abspath(path)}"
            )

    model_config = load_config(config_path)
    Model, _ = load_model(model_config["name"])

    model = Model(model_config)
    load_weight(model, weights_path, strict=False)
    model.to(device)
    model.eval()
    print(f"  [{task_name}] loaded '{model_config['name']}' from {weights_path}")
    return model, model_config


@torch.no_grad()
def batched_predict(
    model,
    tokenizer: EsmTokenizer,
    sequences: List[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
    task_name: str,
) -> np.ndarray:
    """
    Run batched inference and return a (N, out_dim) float32 array.
    Rows for invalid/missing sequences are filled with NaN.
    """
    n_out = len(TASKS[task_name]["output_cols"])
    preds = np.full((len(sequences), n_out), np.nan, dtype=np.float32)

    valid_idx: List[int] = []
    valid_seqs: List[str] = []
    for i, seq in enumerate(sequences):
        norm = normalize_sequence(seq)
        if norm is not None:
            valid_idx.append(i)
            valid_seqs.append(norm)

    if not valid_seqs:
        print(f"  [{task_name}] WARNING: no valid sequences found.")
        return preds

    for start in tqdm(range(0, len(valid_seqs), batch_size),
                      desc=f"  [{task_name}] inferring", unit="batch", leave=False):
        batch_seqs = valid_seqs[start: start + batch_size]
        encoded = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        output = model(input_ids=encoded["input_ids"],
                       attention_mask=encoded["attention_mask"])
        batch_pred = output.pred.detach().cpu().float().numpy()   # (B,) or (B, 2)

        if batch_pred.ndim == 1:
            batch_pred = batch_pred[:, None]   # → (B, 1)

        rows = valid_idx[start: start + len(batch_pred)]
        preds[rows] = batch_pred

    return preds


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # ── Load FASTA ──────────────────────────────────────────
    if not os.path.exists(args.fasta):
        raise FileNotFoundError(f"FASTA file not found: {args.fasta}")
    headers, sequences = read_fasta(args.fasta)
    print(f"Loaded {len(sequences)} sequences from {args.fasta}")

    # ── Build result DataFrame ───────────────────────────────
    accessions = [parse_uniprot_accession(h) for h in headers]
    df = pd.DataFrame({"accession": accessions, "sequence": sequences})

    # ── Run each requested task ──────────────────────────────
    for task_name in args.tasks:
        print(f"\nRunning task: {task_name}")
        model, model_config = load_task_model(task_name, device)

        tokenizer = EsmTokenizer.from_pretrained(model_config["pretrain_model"])
        max_length = int(model_config.get("context_window", 1000))

        preds = batched_predict(
            model=model,
            tokenizer=tokenizer,
            sequences=sequences,
            batch_size=args.batch_size,
            max_length=max_length,
            device=device,
            task_name=task_name,
        )

        for col_idx, col_name in enumerate(TASKS[task_name]["output_cols"]):
            df[col_name] = preds[:, col_idx]

        # free GPU memory between tasks
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Save output ──────────────────────────────────────────
    df.to_csv(args.output, index=False)
    print(f"\nPredictions saved to: {os.path.abspath(args.output)}")

    # ── Quick summary ────────────────────────────────────────
    print("\n── Summary (first 5 rows) ──────────────────────────────")
    cols = ["accession"] + [c for t in args.tasks for c in TASKS[t]["output_cols"]]
    print(df[cols].head().to_string(index=False))


if __name__ == "__main__":
    main()


