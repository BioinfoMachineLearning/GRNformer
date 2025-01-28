"""
author: Akshata 
timestamp: Thu August 24th 2023 11.40 AM
"""
import torch_geometric as pyg
from torch import nn
#import graph_transformer_pytorch as gt

import pandas as pd
import math
import os

import torch
from lightning import Trainer, seed_everything
import torch.nn as nn
import numpy as np
from pandas import read_csv
from torch import  Tensor
from torch.utils.data import DataLoader,ConcatDataset
import wandb
from torch_geometric.utils import negative_sampling
import src.datamodules.grn_dataset_test as dt
#import DatasetMaker.DatasetwithTFcenter_inference_grid as dt
from argparse import ArgumentParser
from typing import Optional, Tuple
from torch_geometric.loader import DataListLoader
from src.models.grnformer.model import GRNFormerLitModule

#from dotenv import load_dotenv

#load_dotenv()
AVAIL_GPUS = [0]
NUM_NODES = 1
BATCH_SIZE = 1
DATALOADERS = 1
ACCELERATOR = "gpu"

DATASET_DIR = os.path.abspath("./")

EPS = 1e-15

"""

torch.set_default_tensor_type(torch.FloatTensor)  # Ensure that the default tensor type is FloatTensor

3
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner to find the best algorithm to use for hardware
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # Set the default tensor type to CUDA FloatTensor
    torch.set_float32_matmul_precision('medium')  # Set Tensor Core precision to medium

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose the device you want to use


if __name__ == "__main__":
    seed_everything(123)
    parser = ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = GRNFormerLinkPred.add_model_specific_args(parser)
    
    parser.add_argument('--exp_file',type=str, default=False,
                        help="sets the expression file of datafolder"
                             "Enter the relative path to the root folder of the dataset eg:'Data/sc-RNA-seq/hESC/hESC_nonspecific_chipseq_500-ExpressionData.csv'" )
    parser.add_argument('--tf_file',type=str,default=False,
                        help="sets the TF file of the data folder format single column CSV file"
                        "Enter the relative path to the transcription factopr file of the species")
    parser.add_argument('--net_file',type=str, default=False,
                        help="sets the ground truth network of datafolder"
                             "Enter the relative path to the root folder of the dataset eg:'Data/sc-RNA-seq/hESC/hESC_nonspecific_chipseq_500-network1.csv'" )
    parser.add_argument('--output_file',type=str, default=False,
                        help="sets the expression file of datafolder"
                             "Enter the relative path to the root folder of the dataset eg:'Data/sc-RNA-seq/hESC/hESC_nonspecific_chipseq_500-ExpressionData.csv'" )
    
    args = parser.parse_args()
    

    root = [os.path.dirname(os.path.abspath(args.exp_file))]
    gene_expression_file=[os.path.abspath(args.exp_file)]
    numnodes= len(pd.read_csv(os.path.abspath(args.exp_file)))
    tffile = os.path.abspath(args.tf_file)
    #if args.TFspecies=="human":
    tf  = pd.read_csv(tffile,header=None)[0].to_list()

    TF_list = [tf]
    regulation_file=[os.path.abspath(args.net_file)]
    #os.makedirs(DATASET_DIR+"/"+args.save_dir, exist_ok=True)
    All_test_dataset=[]
    for i in range(len(root)):

        dataset = dt.GeneExpressionDataset(root[i],gene_expression_file[i],TF_list[i],regulation_file[i])

    All_test_dataset.append(dataset)

    TestDatasets = ConcatDataset(All_test_dataset)
  
    test_loader = DataListLoader(dataset=TestDatasets, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    model = GRNFormerLitModule(totalnodes=numnodes, tf_file = tffile, exp_file = os.path.abspath(args.exp_file), net_file=os.path.abspath(args.net_file), output_file=os.path.abspath(args.output_file))
    print("Model loaded")
    # trainer = pl.Trainer.from_argparse_args(args)
    trainer = Trainer(devices=[0], num_nodes=1, accelerator = ACCELERATOR, detect_anomaly = True, enable_model_summary = True)
    trainer.test(model,dataloaders=test_loader, ckpt_path=os.path.abspath('Trainings/GRNFormer_epoch=26_valid_loss=0.645546.ckpt'))
    #trainer.predict(model,dataloaders=test_loader, ckpt_path=os.path.abspath('Trainings/GRNFormer_epoch=26_valid_loss=0.645546.ckpt'))
