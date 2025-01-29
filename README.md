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

```
git clone https://github.com/BioinfoMachineLearning/GRNformer.git

```

Set up conda environement and install necessary packages using the setup.sh script.

```
cd GRNformer
./setup.sh 
```
## Usage
Run GRNFormer inference on a a sample gene expression file.

```
python infer_grn.py --exp_file /path/to/expression-file.csv --tf_file /path/to/lisoftfs.csv --output_file /path/to/predicted-edges.csv

```
Run GRNFormer to evaluate if ground truth network in present

```
python eval_grn.py --exp_file /path/to/expression-file.csv --tf_file /path/to/lisoftfs.csv --net_file /path/to/ground-truth-network.csv --output_file /path/to/predicted-edges.csv

```

## Evaluate model on test datasets

Download BEELINE sc-RNAseq dataset from the below script.

```
python scripts/collect_data.py --data_dir ./Data/scRNA-seq/ 

```
The downloaded dataset can be found in the Data/scRNA-seq/ and network can be found in Data/scRNA-seq-Networks/ folders


Run evaluation pipeline on test datasets - with all subsets creations

```
python evaluation_pipeline.py --dataset_file Data/mESC.csv --output_dir ./outputs/evluation

```

## Build model from the scratch

Download BEELINE sc-RNAseq dataset from the below script.

```
python scripts/collect_data.py --data_dir ./Data/scRNA-seq/

```
Note: Copying all the Regulatory Networks (Non-specific-Chip-seq-network.csv, STRING-network.csv, [cell-type]-Chip-seq-network.csv) and TFs.csv file to the corresponding cell-type datasets ./Data/scRNA-seq/[cell-type] before begining the training will be of convinience. 

For generalization training, GRNformer combines all the networks for every training dataset.

```
python dataset_combiner.py --cell-type-network ./Data/scRNA-seq/hSEC/hESC-Chip-seq-network.csv --non-specific-network  ./Data/scRNA-seq/hSEC/Non-specific-Chip-seq-network.csv --string-network ./Data/scRNA-seq/hESC/STRING-network.csv --output-file ./Data/scRNA-seq/hESC/hESC-combined.csv
```

Create dataset and splits for training, validation and testing

```
python create_dataset.py --dataset_dir ./Data/sc-RNAseq --dataset_name ./Data/train_list.csv

```

Training the model from scratch

```
python main.py fit --config/grnformer.yaml

```

# Cite Us

@article {Hegde2025.01.26.634966,
	author = {Hegde, Akshata and Cheng, Jianlin},
	title = {GRNFomer: Accurate Gene Regulatory Network Inference Using Graph Transformer},
	elocation-id = {2025.01.26.634966},
	year = {2025},
	doi = {10.1101/2025.01.26.634966},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/01/27/2025.01.26.634966},
	eprint = {https://www.biorxiv.org/content/early/2025/01/27/2025.01.26.634966.full.pdf},
	journal = {bioRxiv}
}
