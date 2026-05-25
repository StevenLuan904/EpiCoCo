# pMHCDiff: Coupling Sequence and Structure Diffusion for pMHC Design

<p align="center">
  <a href="https://icml.cc/Conferences/2026"><img src="https://img.shields.io/badge/ICML-2026-blue.svg" alt="ICML 2026"></a>
</p>

**pMHCDiff** is an SE(3)-equivariant continuous diffusion model for joint peptide sequence and structure co-design, specifically conditioned on MHC pockets. By treating residue mutation and backbone folding as a coupled dynamic process, pMHCDiff achieves biophysical fidelity in *de novo* peptide design.

## 🚀 Overview

pMHCDiff introduces a paradigm shift towards a **Structure-First** generative framework for pMHC (peptide-Major Histocompatibility Complex) binding.
- **Unified Diffusion Architecture**: Couples residue mutation and backbone folding into a single dynamic process.
- **Affinity-Modulated Soft Prompting**: Reinterprets IC50 values as thermodynamic boundary conditions.
- **Bipolar Manifold Steering**: Navigates the generative trajectory using contrastive gradients between high- and low-affinity priors.

---

## 🛠️ Quick Start

### 1. Environment Setup

```bash
# Create and activate the conda environment
conda env create -f env.yaml
conda activate MolDiff
```

### 2. Data Preprocessing

The preprocessing pipeline generates structure and ESM-2 embeddings required for training.

```bash
python preprocess/preprocess.py \
  --input_dir /path/to/pdbs \
  --csv_path /path/to/full_dataset.csv \
  --output_dir ./data/flex_pmhc \
  --threshold 10.0
```

### 3. Training

Update the dataset paths in `configs/train/train_pmhc.yml` before running:

```bash
python scripts/train_pmhc.py \
  --config configs/train/train_pmhc.yml \
  --device cuda:0 \
  --logdir logs
```

### 4. Sampling

Configure the model checkpoint in `configs/sample/sample_pmhc.yml` and run:

```bash
python scripts/sample_pmhc.py \
  --config configs/sample/sample_pmhc.yml \
  --outdir outputs \
  --device cuda:0
```

---

## 📂 Repository Structure

- `configs/`: Training and sampling configurations.
- `models/`: Implementation of the diffusion model and SE(3)-equivariant modules.
- `preprocess/`: Data processing scripts and ESM-2 embedding generation.
- `scripts/`: Entry points for training, sampling, and evaluation.
- `utils/`: Core utilities for dataset handling, evaluation, and visualization.
- `outputs/`: Small example output files for quick inspection.
- `ckpt/`: Placeholder for model checkpoints.

---

## � Example Outputs

Small example outputs are provided in `outputs/sample_pmhc_20260103_113056/` for reference:
- `gen_pep_seq.csv`: Generated peptide sequences.
- `allele_stats_acc.csv`: Per-allele performance statistics.
- `sample_pmhc.yml`: The config used for the generation run.

---

## 💡 Notes

- **ESM Weights**: The preprocessing script downloads ESM weights on the first run via `esm`.
- **Large Datasets**: For large-scale training, set `dataset.force_reload: False` in the config after initial LMDB construction to speed up loading.

---

## �📝 Citation

If you find this work useful, please cite our ICML 2026 paper:

<!-- ```bibtex
@inproceedings{luan2026pmhcdiff,
  title={pMHCDiff: Coupling Sequence and Structure Diffusion for pMHC Design},
  author={Luan, Haoyang and Yu, Gufeng and Chen, Letian and Xiao, Zhenran and Huang, Yueshan and Guo, Junkun and Yang, Yang},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
``` -->
