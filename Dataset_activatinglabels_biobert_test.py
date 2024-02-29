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
import mygene
from sentence_transformers import SentenceTransformer
DATASET_DIR = "/home/aghktb/GRN/GCEN"
test = glob.glob(DATASET_DIR+'/Test/Curated/VSC/**/Expression*.*', recursive=True)



class GCENData(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(test)
    
    
    def __getitem__(self, idx):
        Exp_data = test[idx]
        #Expt_file_name = os.path.basename(pest_data)
        #Read Expression data
        Exp_data_df = read_csv(Exp_data,index_col=0)
        genes = Exp_data_df.index
        #model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        #gsummary = getgenesummary(genes.tolist(),model)
        
        
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

        IsTf = torch.as_tensor(is_tf(genes))
        #is_tf2 = torch.as_tensor(genes.str.fullmatch('|,'.join(TFs.Aliases.values.tolist()),case=False),dtype=torch.bool)
        #tfor = is_tf or is_tf2
        #Generate Z scores for individual expression of a cell across genes
        exp_mat_forz = torch.as_tensor(Exp_data_df.T.to_numpy())
        
        Z_exp_acrossgenes = z_score(exp_mat_forz)
        #norm_pest_intensity_tensor = (pest_intensity_tensor - pest_intensity_tensor.min()) / (pest_intensity_tensor.max() - pest_intensity_tensor.min())
        #generate Z scores of individual expression of a gene across cells
        Z_exp_acrosscell = z_score_dim1(exp_mat_forz)
        return [Z_exp_acrossgenes,IsTf,Exp_tar_adj_ten,Exp_coe_data_ten,Exp_coe_data_mean,Z_exp_acrosscell,exp_mat_forz]#,gsummary]



def encode_paragraph(paragraph,model):
    sentences = paragraph.split(".")
    
    embeddings = model.encode(paragraph)
    #embeddings = np.mean(embeddings,axis=0,keepdims=True)
    return embeddings
def istf(genes):
    #Read human TFs list,and add a bool feature if a gene is a TF or not
    TFs = read_csv(DATASET_DIR+"/Data/Curated/Human_genes.csv",delimiter="\t")
    print(TFs)
        
    is_tf = genes.str.fullmatch('|'.join(TFs.Symbol.values.tolist()),case=False)
def clean_string(s):
    # Remove special characters and convert to lowercase
    s = re.sub(r'[^a-zA-Z0-9]', '', s)
    return s.lower()
def is_tf(genes):
    # Read human TFs list and add a bool feature if a gene is a TF or not
    # Replace with the actual path to your dataset
    TFs = read_csv(DATASET_DIR + "/Data/Curated/Human_genes.csv", delimiter="\t")
    
    # Create a set of all TF symbols and aliases in lower case for case-insensitive matching
    tf_set = set()

    tf_set.update(map(clean_string, TFs['Symbol']))
    TFs = TFs.dropna(subset=['Aliases'])
    for aliases in TFs['Aliases'].str.split(','):
        
        aliases = map(clean_string, filter(None, map(str.strip, aliases)))
        tf_set.update(aliases)
    
    # Check if each gene is a TF or not
    is_tf = genes.str.lower().isin(tf_set)
    
    return is_tf 
     

def getgenesummary(genelist,model):
    mg = mygene.MyGeneInfo()
    gene_ense=dict()
    for gene in genelist:
        result = mg.query(gene, scopes="symbol", fields=["summary"], species="human", verbose=False)
        
        if not result["hits"]:
                 gene_ense[gene]=encode_paragraph("Unknown summary.",model)
        for hit in result["hits"]:
             
            if "summary" in hit:
                gene_ense[gene]=encode_paragraph(hit["summary"],model)
            else:
                  gene_ense[gene]=encode_paragraph("Unknown summary.",model)
           
                   
    return gene_ense


def z_score(exp_mat):
            #exp_mat = torch.log(expression_mat)
            mean_sam = torch.mean(exp_mat, axis = 1,keepdim=True)
            std_sam = torch.mean(exp_mat, axis = 1,keepdim=True)
            
            z_score = (exp_mat - mean_sam)/std_sam
            
            return z_score
def z_score_dim1(exp_mat):
            exp_mat = exp_mat
            
            mean_sam = torch.mean(exp_mat, axis = 0,keepdim=True)
            std_sam = torch.std(exp_mat, axis = 0,keepdim=True)
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
def activate_grn(exp_mat,exp_mean, target_adj,is_tf):
       
       for i in range(len(target_adj)):
              
              if ((exp_mat[i] < exp_mean[i]) and (is_tf[i]==1)):
                     
                     target_adj[i,:] = 0
                     
       return target_adj

class GCENgraph(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.GCENset =[]
        datas = GCENData(DATASET_DIR)
        
        for data in datas:
                
                Zscores_genes,is_tf,Target_grn,Coexp_mat,coexp_mean,zscores_cells,expr_mat = data
                #gsum_ten = torch.as_tensor(np.array(list(gsum.values())))
                #print(gsum_ten)
                gcengraph = construct_networkx(Coexp_mat)
                
                mean = torch.mean(expr_mat,axis=0,keepdim=True)
                
                
                
                for i in range(0,len(Zscores_genes)):
                  node_att = torch.column_stack((is_tf,Zscores_genes[i],zscores_cells[i],coexp_mean))
                  targ_grn_activated = activate_grn(expr_mat[i],mean.T,Target_grn,is_tf.T)
                  positive_edges = coo_tensor_create_pos(targ_grn_activated)
                  neg_edges = coo_tensor_create_neg(targ_grn_activated)
                  gcengraph.x = node_att
                  gcengraph.y = targ_grn_activated
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




