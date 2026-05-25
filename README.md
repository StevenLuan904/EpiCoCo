# EpiCoCo: De Novo Epitope Generation via MHC-Context Co-Modeling and Contrastive Affinity Guidance [ICML 2026]

<p align="center">
  <a href="https://icml.cc/Conferences/2026"><img src="https://img.shields.io/badge/ICML-2026-blue.svg" alt="ICML 2026"></a>
</p>

The *de novo* generation of high-affinity epitopes tailored to specific major histocompatibility complex (MHC) proteins is a pivotal challenge in computational immunotherapy. However, current methods struggle to effectively integrate the MHC context into the generation process, and often fail to guarantee high binding affinity due to the neglect of discriminative signals from non-binders. To bridge these gaps, we present **EpiCoCo**, a probabilistic framework for **Epi**tope generation via MHC-context **Co**-modeling and **Co**ntrastive affinity learning. EpiCoCo treats the pMHC complex as a dynamic, co-adaptive system by operating on the joint E(3) graph. In addition, we introduce Contrastive Affinity Guidance (CAG), an inference mechanism that leverages the gradient difference between learned high- and low-affinity distributions. CAG actively drives the generation trajectory towards high-affinity manifolds while utilizing repulsive signals to filter out candidates with poor binding potential. Extensive evaluations demonstrate that EpiCoCo achieves a mean binding free energy of -45.20 REU, a 23% improvement over the state-of-the-art, while maintaining high structural plausibility. The results validate that context co-modeling and negative-informed guidance are essential for generating valid, high-potency immunotherapeutics.

## 🚀 Overview

EpiCoCo introduces a paradigm shift towards a **Structure-First** generative framework for pMHC (peptide-Major Histocompatibility Complex) binding.
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
  --device cuda:0
```

---

## 📂 Repository Structure

- `configs/`: Training and sampling configurations.
- `models/`: Implementation of the diffusion model and SE(3)-equivariant modules.
- `preprocess/`: Data processing scripts and ESM-2 embedding generation.
- `scripts/`: Entry points for training, sampling, and evaluation.
- `utils/`: Core utilities for dataset handling, evaluation, and visualization.
- `ckpt/`: Placeholder for model checkpoints.