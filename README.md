# PatchET: Learning Enzyme Temperature Properties through Patch-based Neural Architectures

![Architecture](PatchET.png)

**PatchET** is a deep learning model designed to predict enzyme temperature properties:

| Task | Description | Output columns |
|------|-------------|----------------|
| `opt` | Temperature Optimum | `topt` |
| `stability` | Temperature Stability | `t_stability` |
| `range` | Temperature Range | `t_low`, `t_high` |

---

## 🔧 Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔬 Inference

Use `inference.py` to predict enzyme temperature properties from a FASTA file.
Example FASTA files for each task are provided in the `examples/` directory.

### Model weights

Inference needs two sets of weights. `inference.py` downloads whichever are missing the first time it runs, so normally there is nothing to do:

- the [ESM-2 (esm2_t30_150M_UR50D)](https://huggingface.co/facebook/esm2_t30_150M_UR50D) backbone, from the Hugging Face Hub into `esm150/`
- the [PatchET model weights](https://doi.org/10.5281/zenodo.23160814) for the requested task(s), from Zenodo into `checkpoint/<task>/`

Files that are already present are reused. The ESM-2 backbone is frozen during training, so the task checkpoints keep only the PatchET weights: any ESM-2 `pretrain_model` tensors are removed, and the backbone is always loaded from `esm150/`.

To fetch everything in advance (e.g. before working offline), run:

```bash
python download.py                # ESM-2 + all three task checkpoints
python download.py --tasks opt    # ESM-2 + the opt checkpoint only
```

#### Manual download (if the automatic download fails)

If your machine cannot reach Hugging Face or Zenodo, download the files elsewhere and copy them into the repository:

1. **ESM-2 backbone:** download `config.json`, `model.safetensors`, `special_tokens_map.json`, `tokenizer_config.json` and `vocab.txt` from [facebook/esm2_t30_150M_UR50D](https://huggingface.co/facebook/esm2_t30_150M_UR50D/tree/main) into `esm150/`.
2. **PatchET checkpoints:** download `checkpoint.zip` from [Zenodo](https://doi.org/10.5281/zenodo.23160814) and unzip it in the repository root (`unzip checkpoint.zip`), which creates the `checkpoint/` folder.

Either way, the folders should end up like this:

```
esm150/
├── config.json
├── model.safetensors
├── special_tokens_map.json
├── tokenizer_config.json
└── vocab.txt

checkpoint/
├── opt/
│   ├── model_config.yaml
│   └── model.safetensors
├── range/
│   ├── model_config.yaml
│   └── model.safetensors
└── stability/
    ├── model_config.yaml
    └── model.safetensors
```

Run `inference.py` with `--no_download` to use only local files and report anything missing instead of downloading it.

### Basic usage

```bash
python inference.py --fasta <input.fasta> [--tasks TASK ...] [--output OUTPUT] [--batch_size N] [--device DEVICE] [--no_download]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--fasta` | Path to the input FASTA file (required) | — |
| `--tasks` | Task(s) to run: `opt`, `stability`, `range` | `opt` |
| `--output` | Path to the output CSV file | `predictions.csv` |
| `--batch_size` | Batch size for inference | `16` |
| `--device` | Device to use: `auto`, `cpu`, `cuda` | `auto` |
| `--checkpoint_dir` | Folder holding the task checkpoints (one subfolder per task) | `checkpoint/` |
| `--esm_dir` | Folder holding the ESM-2 backbone | `esm150/` |
| `--zenodo_record` | Zenodo record the task checkpoints are downloaded from | `23160814` |
| `--no_download` | Fail instead of downloading missing weights | off |

### Predict Temperature Optimum

```bash
python inference.py --fasta examples/Topt_example.fasta --tasks opt --output topt_predictions.csv
```

### Predict Temperature Stability

```bash
python inference.py --fasta examples/Stability_example.fasta --tasks stability --output stability_predictions.csv
```

### Predict Temperature Range

```bash
python inference.py --fasta examples/Range_example.fasta --tasks range --output range_predictions.csv
```

### Predict all three properties at once

```bash
python inference.py --fasta examples/Topt_example.fasta --tasks opt stability range --output all_predictions.csv
```

### Output format

The output CSV contains the following columns:

| Column | Description |
|--------|-------------|
| `accession` | UniProt accession parsed from the FASTA header |
| `sequence` | Input protein sequence |
| `topt` | Predicted optimal temperature (if `opt` task is run) |
| `t_stability` | Predicted thermostability (if `stability` task is run) |
| `t_low` | Predicted lower bound of active range (if `range` task is run) |
| `t_high` | Predicted upper bound of active range (if `range` task is run) |

---

## 🏋️ Training

Training uses the same frozen ESM-2 backbone. `train.py` downloads it into `esm150/` on the first run if it is missing (see [Manual download](#manual-download-if-the-automatic-download-fails) if that fails). The PatchET checkpoints are not needed for training.

Train the model for each task using the appropriate config file:

**Temperature optimum**
```bash
python train.py \
  --run_config run_configs/opt.yaml \
  --model_config model_configs/PatchET.yaml
```

**Temperature stability**
```bash
python train.py \
  --run_config run_configs/stability.yaml \
  --model_config model_configs/PatchET.yaml
```

**Temperature range**
```bash
python train.py \
  --run_config run_configs/range.yaml \
  --model_config model_configs/PatchET_range.yaml
```

---

## 📄 Citation

If you find PatchET useful in your research, please cite our paper:

```bibtex
@article{Zhang_Yang_Cao_Deng_2026,
  title     = {PatchET: Learning Enzyme Temperature Properties Through Patch-Based Neural Architectures},
  volume    = {40},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/40099},
  DOI       = {10.1609/aaai.v40i34.40099},
  number    = {34},
  journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  author    = {Zhang, Ziqi and Yang, Runze and Cao, Longbing and Deng, Zhaohong},
  year      = {2026},
  month     = {Mar.},
  pages     = {28671--28679}
}
```

