import math
import os
import glob
import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch.nn as nn
import numpy as np
from pandas import read_csv, DataFrame,crosstab
from torch import  Tensor
import torch.nn.functional as F
import re
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import wandb
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryConfusionMatrix, BinaryF1Score,MulticlassAccuracy,MulticlassPrecision,MulticlassRecall,MulticlassF1Score,MulticlassConfusionMatrix
from torchmetrics.regression import MeanSquaredError,R2Score,MeanAbsoluteError
from torch_geometric.data import Data

from argparse import ArgumentParser


DATASET_DIR = "/home/aghktb/GRN/GCEN"
train = glob.glob(DATASET_DIR+'/Data/Curated/GSD/**/Expression*.*', recursive=True)



class GCENData(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(train)
    
    
    def __getitem__(self, idx):
        Exp_data = train[idx]
        #Expt_file_name = os.path.basename(pest_data)
        #Read Expression data
        Exp_data_df = read_csv(Exp_data,index_col=0)
        genes = Exp_data_df.index
        
        #Read the reference network
        Exp_label_pth = os.path.dirname(Exp_data)+"/refNetwork.csv"
        Exp_label_df = read_csv(Exp_label_pth)
        #Here +/- relation is considered  just as an existing regulatory relation i.e 1
        Exp_tar_net = Exp_label_df.replace("+",1)
        Exp_tar_net = Exp_tar_net.replace("-",1)
        Exp_tar_net = Exp_tar_net.drop_duplicates()
        #Create target edge matrix
        Exp_tar_adj = crosstab(Exp_tar_net.Gene1, Exp_tar_net.Gene2)
        idx = Exp_tar_adj.columns.union(Exp_tar_adj.index)
        Exp_tar_adj = Exp_tar_adj.reindex(index = idx, columns=idx, fill_value=0)
        Exp_tar_adj_ten = torch.as_tensor(Exp_tar_adj.to_numpy())
        
        #Apply asin transform on expression data
        Exp_data_df = Exp_data_df.apply(np.arcsinh)
        #Generate paiwaise pearson coorelation of genes
        Exp_coe_data_np =  np.array(np.corrcoef(Exp_data_df.values))
        #Exp_coe_data_np[np.diag_indices_from(Exp_coe_data_np)] = 0
        Exp_coe_data_ten = torch.as_tensor(Exp_coe_data_np)
        #Calulate the average co expression of a gene aganst other genes
        Exp_coe_data_mean = Exp_coe_data_ten.mean(dim=1,keepdim=True)

        #Read human TFs list,and add a bool feature if a gene is a TF or not
        TFs = read_csv(DATASET_DIR+"/Data/Curated/human-tfs.csv")
        is_tf = torch.as_tensor(genes.isin(TFs.TF.values),dtype=torch.bool)
        
        #Generate Z scores for individual expression of a cell across genes
        exp_mat_forz = torch.as_tensor(Exp_data_df.T.to_numpy())
        
        Z_exp_acrossgenes = z_score(exp_mat_forz)
        #norm_pest_intensity_tensor = (pest_intensity_tensor - pest_intensity_tensor.min()) / (pest_intensity_tensor.max() - pest_intensity_tensor.min())
        #generate Z scores of individual expression of a gene across cells
        Z_exp_acrosscell = z_score_dim1(exp_mat_forz)
        return [Z_exp_acrossgenes,is_tf,Exp_tar_adj_ten,Exp_coe_data_ten,Exp_coe_data_mean,Z_exp_acrosscell]

def z_score(exp_mat):
            #exp_mat = torch.log(expression_mat)
            mean_sam = torch.mean(exp_mat, axis = 1,keepdim=True)
            std_sam = torch.mean(exp_mat, axis = 1,keepdim=True)
            
            z_score = (exp_mat - mean_sam)/std_sam
            #print(z_score.shape)
            return z_score
def z_score_dim1(exp_mat):
            exp_mat = exp_mat
            
            mean_sam = torch.mean(exp_mat, axis = 1,keepdim=True)
            std_sam = torch.std(exp_mat, axis = 1,keepdim=True)
            #print(mean_sam.shape)
            z_score = (exp_mat - mean_sam)/std_sam
           
            return z_score
def construct_networkx(edge_weight_matrix):
            """
              Constructs a network from an edge weight matrix.

              Args:
                edge_weight_matrix: A square matrix of edge weights.

              Returns:
                A `torch_geometric.data.Data` object representing the network.
            """

            edge_index = (abs(edge_weight_matrix) > 0.5).nonzero().t()
            row, col = edge_index
            edge_weight = edge_weight_matrix[row, col]
            #G = nx.Graph(np.matrix(edge_weight_matrix))


            # Create a Data object to represent the graph.
            data = Data(edge_index=edge_index, edge_weight=edge_weight)


              # Return the data object.
            return data
def coo_tensor_create_pos(edge_weight_matrix):
    
            edge_index = (edge_weight_matrix > 0).nonzero().t()
            #row, col = edge_index
            #edge_weight = edge_weight_matrix[row, col]
            #coo_ten = to_s(edge_index)
            return edge_index
def coo_tensor_create_neg(edge_weight_matrix):
    
            edge_index = (edge_weight_matrix == 0).nonzero().t()
            #row, col = edge_index
            #edge_weight = edge_weight_matrix[row, col]
            #coo_ten = to_s(edge_index)
            return edge_index
class GCENgraph(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.GCENset =[]
        datas = GCENData(DATASET_DIR)
        
        for data in datas:
                
                Zscores_genes,is_tf,Target_grn,Coexp_mat,coexp_mean,zscores_cells = data
                
                gcengraph = construct_networkx(Coexp_mat)
                positive_edges = coo_tensor_create_pos(Target_grn)
                neg_edges = coo_tensor_create_neg(Target_grn)
                for i in range(0,len(Zscores_genes)):
                  node_att = torch.column_stack((is_tf,Zscores_genes[i],zscores_cells[i],coexp_mean))
                  gcengraph.x = node_att
                  gcengraph.y = Target_grn
                  gcengraph.edges = Coexp_mat
                  gcengraph.pos_edge_label_index = positive_edges
                  gcengraph.neg_edge_label_index = neg_edges
                  self.GCENset.append(gcengraph)
    def __len__(self):
        return len(self.GCENset)
    
    
    def __getitem__(self, idx):
          data = self.GCENset[idx].x
          
          edges = self.GCENset[idx].edge_index
          pos_edges = self.GCENset[idx].pos_edge_label_index
          neg_edges = self.GCENset[idx].neg_edge_label_index
          targ_mat = self.GCENset[idx].y
          edge_attr = self.GCENset[idx].edge_weight.float()
          co_epmat = self.GCENset[idx].edges
          #item = next(GCENdataset)
          return [data,edges, pos_edges,neg_edges,targ_mat,edge_attr,co_epmat]   
          #item = self.GCENset[idx]
          #return item

