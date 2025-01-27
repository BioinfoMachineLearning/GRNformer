# GRNFormer -  Accurate Gene Regulatory Network Inference Using Graph Transformer
GRNFormer, is an advanced variaional graph transformer autoencoder model designed to accurately infer regulatory relationships between transcription factors and target genes from single-cell RNA-seq transcriptomics data, while supporting generalization across species and cell types.

![GRNFormer](./grnformer_overview.png?raw=true "The Overview of GRNFormer Pipeline")

GRNFormer consists of three main novel designs: 
1. TFWalker: a de-novo Transcription Factor (TF) centered subgraph sampling method to extract local or neighborhood co-expression of a transcription factor (TF) to facilitate GRN inference; 
2. End-to-End Learning: 
    -  Gen-Transcoder: a transformer encoder representation module for encoding single-cell RNA-seq (scRNA-seq) gene expression data across different species and cell types
    - a graph transformer model with a GRNFormer Encoder and a variational GRNFormer decoder coupled with GRN inference module for the reconstruction of GRNs; 
3. A novel inference strategy, to incorporate both node features and edge features to infer GRNs for given gene expression data of any given length.

Given a scRNA-seq dataset, a gene co-expression network is first constructed, 
from which a set of subgraphs are sampled by TF-Walker. The subgraphs are processed by GeneTranscoder to generate node and edge embeddings, which are fed to the variational graph transformer autoencoder to learn a GRN representation. The representation is used to infer a gene regulatory sub-network for each subgraph.  The subnetworks are aggregated to construct a full GRN.

The repository contains codes and scripts to create datasets, train the model, evaluate and infer gene regulatory networks.

## Installation

To use this repository, clone this repository to the required folder on your system using

`[git clone https://github.com/BioinfoMachineLearning/GRNformer.git)`

Set up conda environement and install necessary packages using the setup.sh script.

```
cd GRNformer
./setup.sh 
```
## Usage
Run GRNFormer inference on a a sample gene expression file.

```
python inference.py --expFile /path/to/expression-file.csv --tfFile /path/to/lisoftfs.csv --outPrefix /path/to/output-prefix

```
Run GRNFormer to evaluate if ground truth network in present

```
python main.py test --config/grnformer.yaml

```

## Evaluate model on test datasets

Download BEELINE sc-RNAseq dataset from the below script.

```
python scripts/collect_data.py --scrnaseq --data ./Data/sc-RNAseq/ --blindtest

```

Run evaluation pipeline on test datasets - with all subsets creations

```
python evaluation_pipeline.py --dataset Data/mESC.csv

```

## Build model from the scratch

Download BEELINE sc-RNAseq dataset from the below script.

```
python scripts/collect_data.py --scrnaseq --data ./Data/sc-RNAseq/

```

Create dataset and splits

```
python create_dataset.py --dataset ./Data/sc-RNAseq

```

Training the model from scratch

```
python main.py fit --config/grnformer.yaml

```

# Cite Us

