# GRNFormer - Accurate Gene Regulatory Network Inference Using Graph Transformer
[![DOI](https://zenodo.org/badge/1170957484.svg)](https://doi.org/10.5281/zenodo.18868394)

GRNFormer is an advanced variational graph transformer autoencoder model designed to accurately infer regulatory relationships between transcription factors (TFs) and target genes from single-cell RNA-seq transcriptomics data, while supporting generalization across species and cell types.

![GRNFormer](./GRNFormer_overview.png?raw=true "The Overview of GRNFormer Pipeline")

## Overview

GRNFormer consists of three main novel designs:

1. **TFWalker**: A de-novo Transcription Factor (TF) centered subgraph sampling method to extract local or neighborhood co-expression of a transcription factor (TF) to facilitate GRN inference.

2. **End-to-End Learning**:
   - **GeneTranscoder**: A transformer encoder representation module for encoding single-cell RNA-seq (scRNA-seq) gene expression data across different species and cell types.
   - A graph transformer model with a GRNFormer Encoder and a variational GRNFormer decoder coupled with GRN inference module for the reconstruction of GRNs.

3. **Novel Inference Strategy**: Incorporates both node features and edge features to infer GRNs for given gene expression data of any given length.

### Pipeline

Given a scRNA-seq dataset, a gene co-expression network is first constructed, from which a set of subgraphs are sampled by TF-Walker. The subgraphs are processed by GeneTranscoder to generate node and edge embeddings, which are fed to the variational graph transformer autoencoder to learn a GRN representation. The representation is used to infer a gene regulatory sub-network for each subgraph. The subnetworks are aggregated to construct a full GRN.

## Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended for training)
- Conda or Miniconda

### Setup

1. Clone the repository:

```bash
git clone https://github.com/BioinfoMachineLearning/GRNformer.git
cd GRNformer
```

2. Set up conda environment and install necessary packages using the setup script:

```bash
bash setup.sh
```

Alternatively, you can manually create the environment:

```bash
conda env create -f environment.yml
conda activate grnformer
```

## Usage

### Quick Start: Inference on Your Data

Run GRNFormer inference on a sample gene expression file:

```bash
python infer_grn.py \
    --exp_file /path/to/expression-file.csv \
    --tf_file /path/to/listoftfs.csv \
    --output_file /path/to/predicted-edges.csv \
    --coexpression_threshold 0.1 \
    --max_subgraph_size 100
```

**Input File Formats:**
- `expression-file.csv`: Gene expression matrix with genes as rows and cells as columns (or vice versa - the script handles both orientations)
- `listoftfs.csv`: List of transcription factor gene names (one per line or comma-separated)
- `output_file`: Path where the predicted GRN edges will be saved (CSV format: source, target, weight/score)

**Optional Parameters:**
- `--coexpression_threshold` (default: 0.1): Threshold for constructing the co-expression network. Lower values result in denser networks, while higher values create sparser networks.
- `--max_subgraph_size` (default: 100): Maximum number of nodes in each TF-centered subgraph sampled by TFWalker. Adjust based on your dataset size and computational resources.

### Evaluation with Ground Truth
<details>
  <summary>Standard, custom, and general evaluation</summary>

### Standard Evaluation

Run GRNFormer to evaluate performance when a ground truth network is available:

```bash
python eval_grn.py \
    --exp_file /path/to/expression-file.csv \
    --tf_file /path/to/listoftfs.csv \
    --net_file /path/to/ground-truth-network.csv \
    --output_file /path/to/predicted-edges.csv
```
In addition to `predicted-edges.csv` and `predicted-edges-metrics.csv`, the
evaluation also writes `<output_file>_covered_edges.csv`, which contains the
TF→gene edges covered by the TFWalker input (derived from the subgraph
construction). This file can be passed to `scripts/general_grn_evaluation.py`
via `--covered_edges` to ensure only covered edges are evaluated and to compute
coverage.

**Additional Input:**
- `ground-truth-network.csv`: Ground truth network edges (CSV format: source, target)

#### Custom Evaluation with Configurable Parameters

For evaluation with custom coexpression threshold and subgraph size:

```bash
python eval_grn_custom.py \
    --exp_file /path/to/expression-file.csv \
    --tf_file /path/to/listoftfs.csv \
    --net_file /path/to/ground-truth-network.csv \
    --output_file /path/to/predicted-edges.csv \
    --ckpt_path /path/to/checkpoint.ckpt \
    --coexpression_threshold 0.1 \
    --max_subgraph_size 100
```

**Additional Parameters:**
- `--ckpt_path`: Path to the trained model checkpoint file
- `--coexpression_threshold` (default: 0.1): Threshold for co-expression network construction
- `--max_subgraph_size` (default: 100): Maximum subgraph size for TFWalker sampling


### Perturbation Evaluation

Evaluate model robustness under various perturbation conditions (noise and dropout):

**Single test with specific perturbation:**
```bash
python eval_grn_perturb.py \
    --single_test \
    --exp_file /path/to/expression-file.csv \
    --tf_file /path/to/listoftfs.csv \
    --net_file /path/to/ground-truth-network.csv \
    --output_file /path/to/predicted-edges.csv \
    --ckpt_path /path/to/checkpoint.ckpt \
    --noise_std 0.1 \
    --dropout_fraction 0.05 \
    --coexpression_threshold 0.1 \
    --max_subgraph_size 100
```

**Full perturbation sweep** (tests multiple noise and dropout levels):
```bash
python eval_grn_perturb.py \
    --exp_file /path/to/expression-file.csv \
    --tf_file /path/to/listoftfs.csv \
    --net_file /path/to/ground-truth-network.csv \
    --output_file /path/to/predicted-edges.csv \
    --ckpt_path /path/to/checkpoint.ckpt \
    --noise_levels 0.0 0.05 0.1 0.15 0.2 \
    --dropout_levels 0.0 0.05 0.1 0.15 \
    --output_dir ./outputs/perturbation_results \
    --coexpression_threshold 0.1 \
    --max_subgraph_size 100
```

**Perturbation Parameters:**
- `--noise_std`: Standard deviation of Gaussian noise to add to expression data (for single test)
- `--dropout_fraction`: Fraction of genes to randomly drop (for single test)
- `--noise_levels`: Space-separated list of noise levels for sweep (e.g., "0.0 0.05 0.1 0.15 0.2")
- `--dropout_levels`: Space-separated list of dropout fractions for sweep (e.g., "0.0 0.05 0.1 0.15")
- `--absolute_noise`: Use absolute noise values instead of scaled (default: noise is scaled relative to data std)
- `--output_dir`: Directory to save perturbation sweep results
- `--coexpression_threshold` (default: 0.1): Threshold for co-expression network construction
- `--max_subgraph_size` (default: 100): Maximum subgraph size for TFWalker sampling


### Complete GRN Evaluation (clean negative pool, sampling, full-matrix, EPR)

GRNFormer’s complete evaluation proceeds in two stages:

1. **Clean negative pool construction**

   From the expression matrix and ground-truth network, we construct a **clean
   negative evaluation pool**. This pool contains all ordered gene–gene pairs
   `(g1, g2)` with `g1 != g2` in the expression gene set, **excluding**:

   - all known positive TF–target edges from the reference network, and
   - any training negatives you optionally provide.

   This ensures that negatives used for evaluation do not overlap with known
   positives or training negatives.

2. **Metric computation**

   Using the clean negative pool, the ground-truth positives, and the full
   predicted TF–gene adjacency, we compute:

   - sampled AUROC/AUPR (with bootstrapping),
   - full-matrix AUROC/AUPR over the entire clean pool,
   - early precision (EPR@K),
   - coverage of the ground-truth network by the TFWalker subgraphs.

---

#### Step 1: Build the clean negative evaluation pool

Script: `scripts/create_clean_eval_pool.py`

**Purpose**

- Define a clean set of negative TF–gene candidates for evaluation, consistent
  across methods and runs.

**Arguments**

- `--expression`  
  Path to `ExpressionData.csv`. Genes in the index define the gene universe.

- `--network`  
  Path to the reference regulatory network (`refNetwork.csv`). All TF–target
  pairs in this file are treated as positives and excluded from the clean pool.

- `--training_negatives` (optional)  
  One or more CSV files with training negatives (e.g. negatives sampled during
  model training). Any pairs in these files are also excluded from the clean pool.

- `--output`  
  Path to the output CSV, typically named
  `clean_evaluation_pool_all_pairs.csv`. The file contains all remaining TF–gene
  candidate pairs and is used as the negative universe for evaluation.

**Example**
```bash
python scripts/create_clean_eval_pool.py \
  --expression /path/to/ExpressionData.csv \
  --network /path/to/refNetwork.csv \
  --output /path/to/clean_evaluation_pool_all_pairs.csv
```

#### Step 2: Run the general GRN evaluation

Script: `scripts/general_grn_evaluation.py`

**Purpose**

Evaluate GRNFormer predictions against the ground-truth regulatory network
using the clean negative pool and TFWalker coverage.

**Inputs**

- `--positives`  
  Ground-truth regulatory network (e.g. `refNetwork.csv` or `master_test.csv`).
  If a `label` / `Label` column exists, only `label == 1` rows are used.

- `--clean_negatives`  
  Clean negative pool from Step 1 (e.g. `clean_evaluation_pool_all_pairs.csv`).

- `--predictions`  
  Full TF–gene adjacency with prediction scores (e.g. `predictedNetwork.csv`),
  as produced by `eval_grn.py`.

- `--expression`  
  Expression matrix (`ExpressionData.csv`, genes in the index). This defines the
  gene universe and filters positives/negatives/predictions.

- `--tfs`  
  TF list (`TFs.csv`). Positives are restricted to TF→gene edges where the
  source is in this TF list and in the expression gene set.

- `--covered_edges` (optional but recommended)  
  CSV listing TF→gene edges covered by the TFWalker subgraphs
  (e.g. `Gene1,Gene2`, derived from `edge_index_unique`). This encodes which
  ground-truth TF→gene interactions are reachable in the TF-centered subgraphs
  and is used to restrict evaluation to covered edges and to compute coverage.

- `--sampled_neg_ratio`  
  Ratio of sampled negatives to positives for sampled evaluation (default 1.0).

- `--epr_k`  
  Comma-separated K values for EPR@K (default: K = number of positives).

- `--output_json`  
  Path to save all metrics in JSON format.

**Example**
```bash
python scripts/general_grn_evaluation.py \
  --positives /path/to/refNetwork.csv \
  --clean_negatives /path/to/clean_evaluation_pool_all_pairs.csv \
  --predictions /path/to/predictedNetwork.csv \
  --expression /path/to/ExpressionData.csv \
  --tfs /path/to/TFs.csv \
  --covered_edges /path/to/predictedNetwork_covered_edges.csv \
  --sampled_neg_ratio 1.0 \
  --epr_k 10,50,100 \
  --output_json /path/to/metrics.json
```

**Outputs**

The JSON produced by `--output_json` contains the following key fields:

- **Counts**
  - `total_positives_in_file`  
    Number of TF→gene positives in the ground-truth file after TF/expression filtering.
  - `n_positives_with_predictions`  
    Number of positives actually evaluated (after intersecting with
    `--covered_edges`, if provided).
  - `positive_coverage`  
    Fraction of ground-truth TF→gene edges covered by the TFWalker subgraphs:  
    `n_positives_with_predictions / total_positives_in_file`.
  - `n_full_negatives`  
    Size of the clean negative pool.
  - `n_sampled_negatives`  
    Number of negatives used in each sampled evaluation run.

- **Sampled metrics (per-run and bootstrapped)**
  - `sampled_auroc`, `sampled_aupr`  
    AUROC and AUPR for a single sampled negative set.
  - `sampled_auroc_mean`, `sampled_auroc_std`  
    Mean and standard deviation of sampled AUROC over 100 bootstrap repeats.
  - `sampled_aupr_mean`, `sampled_aupr_std`  
    Mean and standard deviation of sampled AUPR (average precision) over 100
    bootstrap repeats.

- **Full-matrix metrics**
  - `full_auroc`, `full_aupr`  
    AUROC and AUPR computed using all positives vs. all negatives in the clean
    evaluation pool.

- **Early Precision (EPR)**
  - `epr@K`  
    Early precision values at the K values specified via `--epr_k` (plus
    `K = number of positives` if not already included).

</details>

## Evaluation on Test Datasets
<details>
  <summary>Click to see the details</summary>

### Download BEELINE Datasets

Download BEELINE sc-RNAseq datasets:

```bash
python collect_data.py --data_dir ./Data/scRNA-seq/
```

The downloaded datasets can be found in:
- `Data/scRNA-seq/` - Expression data
- `Data/scRNA-seq-Networks/` - Network data

### Run Evaluation Pipeline

Run the evaluation pipeline on test datasets with all subset creations:

```bash
python evaluation_pipeline.py \
    --dataset_file Data/mESC.csv \
    --output_dir ./outputs/evaluation
```
</details>

## Training from Scratch
<details>
  <summary>Click to see the details</summary>
  
### 1. Prepare Datasets

Download BEELINE sc-RNAseq datasets:

```bash
python collect_data.py --data_dir ./Data/scRNA-seq/
```

**Note:** Before beginning training, copy all the Regulatory Networks (Non-specific-Chip-seq-network.csv, STRING-network.csv, [cell-type]-Chip-seq-network.csv) and TFs.csv file to the corresponding cell-type datasets in `./Data/scRNA-seq/[cell-type]/`.

### 2. Combine Networks

For generalization training, GRNformer combines all the networks for every training dataset:

```bash
python dataset_combiner.py \
    --cell-type-network ./Data/scRNA-seq/hESC/hESC-Chip-seq-network.csv \
    --non-specific-network ./Data/scRNA-seq/hESC/Non-specific-Chip-seq-network.csv \
    --string-network ./Data/scRNA-seq/hESC/STRING-network.csv \
    --output-file ./Data/scRNA-seq/hESC/hESC-combined.csv
```

### 3. Create Dataset Splits

Create dataset and splits for training, validation, and testing:

```bash
python create_dataset.py \
    --dataset_dir ./Data/sc-RNAseq \
    --dataset_name ./Data/train_list.csv
```

### 4. Train the Model

Train the model from scratch using the configuration file:

```bash
python main.py fit --config config/grnformer.yaml
```

You can customize training parameters by editing `config/grnformer.yaml` or by passing command-line arguments.
</details>

## Datasets

### Available Datasets

- **BEELINE**: https://zenodo.org/records/3701939
- **DREAM5**: https://www.synapse.org/Synapse:syn2787209/wiki/70351
- **PBMC3k**: https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc3k
- **Preprocessed PBMC**: Can be accessed from the `scanpy` Python package

## Project Structure

```
GRNformer/
├── src/
│   ├── models/
│   │   └── grnformer/
│   │       ├── model.py          # Main GRNFormer model
│   │       └── network.py        # Network architecture
│   └── datamodules/
│       ├── grn_datamodule.py     # Training data module
│       ├── grn_dataset_inference.py  # Inference dataset
│       └── grn_dataset_test.py   # Test dataset
├── config/
│   └── grnformer.yaml            # Training configuration
├── main.py                       # Training entry point
├── infer_grn.py                  # Inference script
├── eval_grn.py                        # Standard evaluation script
├── eval_grn_custom.py                 # Custom evaluation with configurable parameters
├── eval_grn_perturb.py                # Perturbation evaluation script
├── scripts/general_grn_evaluation.py  # General GRN evaluation (sampled/full AUROC/AUPR, EPR, coverage)
├── scripts/create_clean_eval_pool.py  # Clean negative pool construction
├── evaluation_pipeline.py             # Full evaluation pipeline
├── create_dataset.py                  # Dataset creation
├── dataset_combiner.py                # Network combination
├── collect_data.py                    # Data download
└── environment.yml                    # Conda environment
```

## Citation

If you use GRNFormer in your research, please cite:

```bibtex
@article {Hegde2025.01.26.634966,
	author = {Hegde, Akshata and Cheng, Jianlin},
	title = {GRNFormer: Accurate Gene Regulatory Network Inference Using Graph Transformer},
	elocation-id = {2025.01.26.634966},
	year = {2025},
	doi = {10.1101/2025.01.26.634966},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/01/27/2025.01.26.634966},
	eprint = {https://www.biorxiv.org/content/early/2025/01/27/2025.01.26.634966.full.pdf},
	journal = {bioRxiv}
}
```

## License

See [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on the GitHub repository.
