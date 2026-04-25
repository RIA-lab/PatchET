"""
PatchET / PatchEX Inference Script
===================================
Predicts thermal properties for protein sequences from a FASTA file.

Supported tasks and default model backends:
  - opt       : optimal temperature         (PatchET or PatchEX)
  - stability : thermostability threshold   (PatchET only)
  - range     : [T_low, T_high] active range (PatchET only)

Checkpoint directory layout expected under the repo root:

  PatchET/
  ├── opt/        { model_config.yaml, model.safetensors }
  ├── stability/  { model_config.yaml, model.safetensors }
  └── range/      { model_config.yaml, model.safetensors }

  PatchEX/
  └── opt/        { model_config.yaml, model.safetensors }

Usage:
  # PatchET (default)
  python inference.py --fasta proteins.fasta --output predictions.csv --tasks opt stability range

  # PatchEX for opt, with per-residue weight output
  python inference.py --fasta proteins.fasta --model patchex --tasks opt \\
      --output predictions.csv --weights_output weights/

  # Mix: PatchET stability + range, PatchEX opt
  python inference.py --fasta proteins.fasta --model patchex --tasks opt stability range \\
      --output predictions.csv --weights_output weights/
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import EsmTokenizer

from models import load_model
from utils import load_config, load_weight, read_fasta, replace_noncanonical


# ─────────────────────────────────────────────────────────────────────────────
# Task / checkpoint configuration
# ─────────────────────────────────────────────────────────────────────────────

PATCHET_TASKS: Dict[str, Dict] = {
    "opt": {
        "checkpoint_dir": "checkpoint/PatchET/opt",
        "output_cols": ["topt"],
    },
    "stability": {
        "checkpoint_dir": "checkpoint/PatchET/stability",
        "output_cols": ["t_stability"],
    },
    "range": {
        "checkpoint_dir": "checkpoint/PatchET/range",
        "output_cols": ["t_low", "t_high"],
    },
}

# PatchEX is currently only trained for opt and ph; extend here as new tasks become available.
PATCHEX_TASKS: Dict[str, Dict] = {
    "opt": {
        "checkpoint_dir": "checkpoint/PatchEX/opt",
        "output_cols": ["topt"],
    },
    "ph": {
        "checkpoint_dir": "checkpoint/PatchEX/ph",
        "output_cols": ["phopt"],
    },
}

PATCH_LEN = 25          # residues per patch (fixed in both model families)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SeqItem:
    """Holds one sequence and its inference outputs."""
    accession: str
    sequence: str
    label: float = 0.0
    weights: Any = None         # patch-level weights (np.ndarray after map_weights)
    idx: list = field(default_factory=list)
    score: Optional[np.ndarray] = None

    def map_weights(self) -> None:
        """
        Expand patch-level weights to per-residue weights by repeating each
        patch weight across its PATCH_LEN residues.
        After this call, self.weights is a 1-D np.ndarray of length len(sequence).
        """
        length = len(self.sequence)
        last_patch_idx = length // PATCH_LEN
        last_patch_residue = length % PATCH_LEN
        expanded: List[float] = []
        for i in range(last_patch_idx):
            expanded.extend([float(self.weights[i])] * PATCH_LEN)
        if last_patch_residue > 0:
            expanded.extend([float(self.weights[last_patch_idx])] * last_patch_residue)
        self.weights = np.asarray(expanded, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PatchET/PatchEX: predict thermal properties from a FASTA file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--fasta", type=str, required=True,
        help="Path to the input FASTA file.",
    )
    parser.add_argument(
        "--output", type=str, default="predictions.csv",
        help="Path to the output CSV file (default: predictions.csv).",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+",
        choices=["opt", "stability", "range", "ph"], default=["opt"],
        help=(
            "Which property tasks to run (default: opt). "
            "PatchEX supports 'opt' and 'ph'; 'stability' and 'range' use PatchET only."
        ),
    )
    parser.add_argument(
        "--model", type=str, choices=["patchet", "patchex"], default="patchet",
        help=(
            "Model backend to use (default: patchet). "
            "When 'patchex' is chosen, tasks available in PatchEX use the PatchEX checkpoint; "
            "remaining tasks (stability, range) automatically fall back to PatchET."
        ),
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for inference (default: 16).",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--weights_output", type=str, default=None,
        help=(
            "Directory to save per-residue patch-weight files (PatchEX only). "
            "One JSON file is written per sequence: <accession>.json. "
            "If omitted, weights are added as a column in the output CSV instead."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(choice)


def parse_uniprot_accession(header: str) -> str:
    """
    Extract UniProt accession from FASTA header, e.g. 'sp|P12345|ENTRY_NAME ...'.
    Falls back to the first whitespace-delimited token.
    """
    match = re.match(r"^(?:sp|tr)\|([A-Za-z0-9_-]+)\|", header)
    if match:
        return match.group(1)
    return header.split()[0] if header.strip() else header


def normalize_sequence(value: object) -> Optional[str]:
    """Strip whitespace, upper-case, replace non-canonical residues with X."""
    if not isinstance(value, str):
        return None
    seq = re.sub(r"\s+", "", value.strip().upper())
    if not seq:
        return None
    return replace_noncanonical(seq, replace_char="X")


def resolve_checkpoint(task_name: str, use_patchex: bool) -> Tuple[str, str, bool]:
    """
    Return (checkpoint_dir, effective_model_name, is_patchex).
    Falls back to PatchET if the task is not available in PatchEX.
    """
    if task_name in PATCHEX_TASKS:
        if use_patchex:
            cfg = PATCHEX_TASKS[task_name]
            return cfg["checkpoint_dir"], "patchex", True
        elif task_name not in PATCHET_TASKS:
            # Task exists only in PatchEX (e.g. ph) — must use --model patchex
            raise ValueError(
                f"Task '{task_name}' is only supported by PatchEX. "
                f"Please add --model patchex to your command."
            )
    # PatchET (default or fallback for tasks available in both)
    cfg = PATCHET_TASKS[task_name]
    if use_patchex:
        print(
            f"  [{task_name}] PatchEX checkpoint not available for this task — "
            f"falling back to PatchET."
        )
    return cfg["checkpoint_dir"], "patchet", False


def load_task_model(
    task_name: str, use_patchex: bool, device: torch.device
) -> Tuple[Any, Dict, bool]:
    """
    Load model + config for a task. Returns (model, model_config, is_patchex).
    """
    checkpoint_dir, backend, is_patchex = resolve_checkpoint(task_name, use_patchex)
    config_path  = os.path.join(checkpoint_dir, "model_config.yaml")
    weights_path = os.path.join(checkpoint_dir, "model.safetensors")

    for path, label in [(config_path, "config"), (weights_path, "weights")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[{task_name}/{backend}] {label} not found: {os.path.abspath(path)}"
            )

    model_config = load_config(config_path)
    Model, _ = load_model(model_config["name"])

    model = Model(model_config)
    load_weight(model, weights_path, strict=False)
    model.inference = True
    model.to(device)
    model.eval()

    print(f"  [{task_name}] loaded '{model_config['name']}' from {weights_path}")
    return model, model_config, is_patchex


# ─────────────────────────────────────────────────────────────────────────────
# Inference runners
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_patchet(
    model,
    tokenizer: EsmTokenizer,
    sequences: List[str],
    valid_idx: List[int],
    batch_size: int,
    max_length: int,
    device: torch.device,
    task_name: str,
) -> np.ndarray:
    """
    Standard PatchET inference. Returns float32 array of shape (N, n_out),
    with NaN rows for invalid sequences.
    """
    n_out = len(PATCHET_TASKS[task_name]["output_cols"])
    preds = np.full((len(sequences), n_out), np.nan, dtype=np.float32)

    valid_seqs = [sequences[i] for i in valid_idx]
    if not valid_seqs:
        return preds

    for start in tqdm(range(0, len(valid_seqs), batch_size),
                      desc=f"  [{task_name}/patchet] inferring", unit="batch", leave=False):
        batch_seqs = valid_seqs[start: start + batch_size]
        encoded = tokenizer(
            batch_seqs, return_tensors="pt",
            padding="max_length", truncation=True, max_length=max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        output = model(input_ids=encoded["input_ids"],
                       attention_mask=encoded["attention_mask"])
        batch_pred = output.pred.detach().cpu().float().numpy()  # (B,) or (B,2)
        if batch_pred.ndim == 1:
            batch_pred = batch_pred[:, None]

        rows = valid_idx[start: start + len(batch_pred)]
        preds[rows] = batch_pred

    return preds


@torch.no_grad()
def run_patchex(
    model,
    tokenizer: EsmTokenizer,
    seq_items: List[SeqItem],
    valid_idx: List[int],
    batch_size: int,
    max_length: int,
    device: torch.device,
    task_name: str,
) -> None:
    """
    PatchEX inference with patch-weight extraction.
    Fills seq_items[i].label and seq_items[i].weights (patch-level) in-place
    for each valid index, then calls map_weights() to expand to residue level.
    """
    valid_items = [seq_items[i] for i in valid_idx]
    if not valid_items:
        return

    for start in tqdm(range(0, len(valid_items), batch_size),
                      desc=f"  [{task_name}/patchex] inferring", unit="batch", leave=False):
        batch_items = valid_items[start: start + batch_size]
        batch_seqs  = [item.sequence for item in batch_items]

        encoded = tokenizer(
            batch_seqs, return_tensors="pt",
            padding="max_length", truncation=True, max_length=max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        output = model(input_ids=encoded["input_ids"],
                       attention_mask=encoded["attention_mask"])

        # output.pred: [B] or [B,1], output.patch_weights: [B, P]
        preds_cpu   = output.pred.detach().cpu()
        weights_cpu = output.patch_weights.detach().cpu()  # [B, P]

        for local_i, item in enumerate(batch_items):
            pred_val = preds_cpu[local_i]
            item.label   = pred_val.item() if pred_val.numel() == 1 else pred_val[0].item()
            item.weights = weights_cpu[local_i].numpy()   # patch-level, shape [P]
            item.map_weights()                            # expand to residue-level


# ─────────────────────────────────────────────────────────────────────────────
# Weight output helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_weights_to_dir(seq_items: List[SeqItem], output_dir: str, task_name: str) -> None:
    """
    Write one JSON file per sequence into output_dir/<task_name>/.
    Each file contains the accession, sequence, predicted value, and per-residue weights.
    """
    task_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)

    for item in seq_items:
        if item.weights is None:
            continue
        record = {
            "accession":       item.accession,
            "sequence":        item.sequence,
            "predicted_value": round(float(item.label), 4),
            "residue_weights": item.weights.tolist(),
            "patch_weights":   item.weights[::PATCH_LEN].tolist(),  # one value per patch
        }
        safe_name = re.sub(r"[^\w\-.]", "_", item.accession)
        out_path  = os.path.join(task_dir, f"{safe_name}.json")
        with open(out_path, "w") as fh:
            json.dump(record, fh, indent=2)

    print(f"  [{task_name}] per-residue weights saved to: {os.path.abspath(task_dir)}/")


def weights_to_csv_column(seq_items: List[SeqItem]) -> List[Optional[str]]:
    """
    Serialise per-residue weights as a semicolon-separated string for CSV storage.
    Returns None for items without weights.
    """
    out = []
    for item in seq_items:
        if item.weights is not None:
            out.append(";".join(f"{w:.6f}" for w in item.weights))
        else:
            out.append(None)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)
    use_patchex = (args.model == "patchex")

    print(f"Using device : {device}")
    print(f"Model backend: {args.model}")

    # ── Load FASTA ────────────────────────────────────────────────────────────
    if not os.path.exists(args.fasta):
        raise FileNotFoundError(f"FASTA file not found: {args.fasta}")
    headers, sequences = read_fasta(args.fasta)
    print(f"Loaded {len(sequences)} sequences from {args.fasta}")

    accessions = [parse_uniprot_accession(h) for h in headers]

    # Normalise sequences; track which are valid
    norm_sequences: List[Optional[str]] = [normalize_sequence(s) for s in sequences]
    valid_idx: List[int] = [i for i, s in enumerate(norm_sequences) if s is not None]
    if len(valid_idx) < len(sequences):
        print(f"  WARNING: {len(sequences) - len(valid_idx)} sequence(s) skipped (empty / invalid).")

    # ── Build result DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame({"accession": accessions, "sequence": sequences})

    # ── Run each requested task ───────────────────────────────────────────────
    for task_name in args.tasks:
        print(f"\nRunning task: {task_name}")
        model, model_config, task_is_patchex = load_task_model(task_name, use_patchex, device)
        tokenizer  = EsmTokenizer.from_pretrained(model_config["pretrain_model"])
        max_length = int(model_config.get("context_window", 1000))

        valid_seqs_norm = [norm_sequences[i] for i in valid_idx]

        if task_is_patchex:
            # ── PatchEX path: uses SeqItem objects, extracts patch weights ────
            seq_items = [
                SeqItem(accession=accessions[i], sequence=norm_sequences[i])
                for i in valid_idx
            ]
            run_patchex(
                model=model,
                tokenizer=tokenizer,
                seq_items=seq_items,
                valid_idx=list(range(len(seq_items))),  # all items in seq_items are valid
                batch_size=args.batch_size,
                max_length=max_length,
                device=device,
                task_name=task_name,
            )

            # Collect predictions into the DataFrame
            out_col = PATCHEX_TASKS[task_name]["output_cols"][0]
            pred_col = np.full(len(sequences), np.nan, dtype=np.float32)
            for local_i, global_i in enumerate(valid_idx):
                pred_col[global_i] = seq_items[local_i].label
            df[out_col] = pred_col

            # Build full-length seq_items list (with None-weight placeholders for invalids)
            all_items: List[Optional[SeqItem]] = [None] * len(sequences)
            for local_i, global_i in enumerate(valid_idx):
                all_items[global_i] = seq_items[local_i]

            # Build a flat list for weight export (invalid rows have weights=None)
            flat_items = []
            for i, item in enumerate(all_items):
                if item is None:
                    flat_items.append(SeqItem(accession=accessions[i], sequence=sequences[i] or ""))
                else:
                    flat_items.append(item)

            # Save / embed weights
            if args.weights_output:
                save_weights_to_dir(
                    [item for item in flat_items if item.weights is not None],
                    args.weights_output,
                    task_name,
                )
            else:
                df[f"{out_col}_residue_weights"] = weights_to_csv_column(flat_items)
                print(
                    f"  [{task_name}] per-residue weights embedded in CSV column "
                    f"'{out_col}_residue_weights' (semicolon-separated)."
                )

        else:
            # ── PatchET path: simple batched inference ────────────────────────
            preds = run_patchet(
                model=model,
                tokenizer=tokenizer,
                sequences=valid_seqs_norm,          # already-normalised, valid only
                valid_idx=list(range(len(valid_idx))),
                batch_size=args.batch_size,
                max_length=max_length,
                device=device,
                task_name=task_name,
            )

            # preds rows correspond to valid_idx positions; map back to full N
            output_cols = PATCHET_TASKS[task_name]["output_cols"]
            full_preds  = np.full((len(sequences), len(output_cols)), np.nan, dtype=np.float32)
            for local_i, global_i in enumerate(valid_idx):
                full_preds[global_i] = preds[local_i]

            for col_idx, col_name in enumerate(output_cols):
                df[col_name] = full_preds[:, col_idx]

        # Free GPU memory between tasks
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Save output CSV ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nPredictions saved to: {os.path.abspath(args.output)}")

    # ── Quick summary ─────────────────────────────────────────────────────────
    pred_cols = []
    for task_name in args.tasks:
        _, _, task_is_patchex = resolve_checkpoint(task_name, use_patchex)
        if task_is_patchex:
            pred_cols += PATCHEX_TASKS[task_name]["output_cols"]
        else:
            pred_cols += PATCHET_TASKS[task_name]["output_cols"]

    summary_cols = ["accession"] + [c for c in pred_cols if c in df.columns]
    print("\n── Summary (first 5 rows) " + "─" * 45)
    print(df[summary_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()