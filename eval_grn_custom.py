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
import src.datamodules.grn_dataset_test_custom as dt
#import DatasetMaker.DatasetwithTFcenter_inference_grid as dt
from argparse import ArgumentParser
from typing import Optional, Tuple
from torch_geometric.loader import DataListLoader
from src.models.grnformer.model import GRNFormerLitModule

#from dotenv import load_dotenv
import psutil
import time
from contextlib import contextmanager

def get_memory_usage():
    """Get current memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024  # MB
    else:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

def get_peak_memory_usage():
    """Get peak memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        return get_memory_usage()

@contextmanager
def memory_monitor(operation_name):
    """Context manager to monitor memory usage for an operation"""
    start_memory = get_memory_usage()
    start_time = time.time()
    
    print(f"Starting {operation_name} - Memory: {start_memory:.2f} MB")
    
    try:
        yield
    finally:
        end_memory = get_memory_usage()
        peak_memory = get_peak_memory_usage()
        end_time = time.time()
        
        print(f"Completed {operation_name}")
        print(f"  - Time: {end_time - start_time:.2f} seconds")
        print(f"  - Memory before: {start_memory:.2f} MB")
        print(f"  - Memory after: {end_memory:.2f} MB")
        print(f"  - Memory delta: {end_memory - start_memory:.2f} MB")
        print(f"  - Peak memory: {peak_memory:.2f} MB")
        
        # Reset peak memory counter
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
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
    parser.add_argument('--ckpt_path',type=str, default=False,
                        help="sets the checkpoint path"
                             "Enter the relative path to the checkpoint file eg:'Trainings/GRNFormer_epoch=26_valid_loss=0.645546.ckpt'" )
    parser.add_argument('--coexpression_threshold',type=float, default=0.1,
                        help="sets the coexpression threshold"
                             "Enter the coexpression threshold as a float value eg:0.1" )
    parser.add_argument('--max_subgraph_size',type=int, default=100,
                        help="sets the maximum subgraph size"
                             "Enter the maximum subgraph size as an integer value eg:100" )
    args = parser.parse_args()
    
    # Monitor initial memory
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")
    
    # Monitor data loading
    with memory_monitor("Data Loading"):
        root = [os.path.dirname(os.path.abspath(args.exp_file))]
        gene_expression_file=[os.path.abspath(args.exp_file)]
        numnodes= len(pd.read_csv(os.path.abspath(args.exp_file),index_col=0))#.transpose())
        print(numnodes)
        tffile = os.path.abspath(args.tf_file)
        tf  = pd.read_csv(tffile,header=None)[0].to_list()
        TF_list = [tf]
        regulation_file=[os.path.abspath(args.net_file)]
        
        All_test_dataset=[]
        for i in range(len(root)):
                dataset = dt.GeneExpressionDataset(root[i],gene_expression_file[i],TF_list[i],regulation_file[i],coexpression_threshold=args.coexpression_threshold,max_subgraph_size=args.max_subgraph_size)
        All_test_dataset.append(dataset)
        TestDatasets = ConcatDataset(All_test_dataset)
        test_loader = DataListLoader(dataset=TestDatasets, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    
    # Monitor model creation
    with memory_monitor("Model Creation"):
        model = GRNFormerLitModule(
            totalnodes=numnodes, 
            tf_file=tffile, 
            exp_file=os.path.abspath(args.exp_file), 
            net_file=os.path.abspath(args.net_file), 
            output_file=os.path.abspath(args.output_file)
        )
        print("Model loaded")
    
    # Monitor model loading from checkpoint
    with memory_monitor("Model Loading from Checkpoint"):
        trainer = Trainer(devices=[0], num_nodes=1, accelerator=ACCELERATOR, detect_anomaly=True, enable_model_summary=True)
    
    # Monitor inference
    with memory_monitor("Model Inference"):
        trainer.test(model, dataloaders=test_loader, ckpt_path=os.path.abspath(args.ckpt_path))
    
    # Final memory report
    print(f"Final memory usage: {get_memory_usage():.2f} MB")
    print(f"Peak memory usage: {get_peak_memory_usage():.2f} MB")