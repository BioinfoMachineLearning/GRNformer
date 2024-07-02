import math
import os
import glob
import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch.nn as nn
import numpy as np
from pandas import read_csv, DataFrame,crosstab,unique
from torch import  Tensor
import torch.nn.functional as F
import re
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.utils import negative_sampling
import wandb
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryConfusionMatrix, BinaryF1Score,MulticlassAccuracy,MulticlassPrecision,MulticlassRecall,MulticlassF1Score,MulticlassConfusionMatrix
from torchmetrics.regression import MeanSquaredError,R2Score,MeanAbsoluteError
from torch_geometric.data import Data
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.utils import subgraph
from argparse import ArgumentParser
import mygene
from sentence_transformers import SentenceTransformer
import random
DATASET_DIR = "/home/aghktb/GRN/GRNformer"
BATCH_SIZE = 1
train = glob.glob(DATASET_DIR+'/Data/sc-RNA-seq/hHep/Filtered--ExpressionData.csv', recursive=True)



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
        #print(len(genes))
        #model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        #gsummary = getgenesummary(genes.tolist(),model)
        
        
        #Read the reference network
        Exp_label_pth = os.path.dirname(Exp_data)+"/Filtered--network1.csv"
        Exp_label_df = read_csv(Exp_label_pth)
        #Here +/- relation is considered  just as an existing regulatory relation i.e 1
        #Exp_tar_net = Exp_label_df.replace("+",1)
        #Exp_tar_net = Exp_tar_net.replace("-",1)
        #Exp_tar_net = Exp_tar_net.drop_duplicates()
        #print(len(Exp_tar_net))
        #Create target edge matrix
        #Exp_tar_adj = crosstab(Exp_label_df.Gene1, Exp_label_df.Gene2)
        #Exp_tar_adj = crosstab(genes,genes)
        #print(Exp_tar_adj.shape)
        #idx = Exp_tar_adj.columns.union(Exp_tar_adj.index)
        #Exp_tar_adj = Exp_tar_adj.reindex(index = idx, columns=idx, fill_value=0)
        #Exp_tar_adj_ten = torch.as_tensor(Exp_tar_adj.to_numpy())
        #print(Exp_tar_adj_ten)
        genes_tar = unique(Exp_label_df[['Gene1', 'Gene2']].values.ravel('K'))
        Exp_tar_adj = DataFrame(index=genes, columns=genes)

        # Fill in the values based on df2
        for index, row in Exp_label_df.iterrows():
            Exp_tar_adj.at[row['Gene1'], row['Gene2']] = row['Type']
        Exp_tar_adj = Exp_tar_adj.fillna(0)
        Exp_tar_adj_ten = torch.as_tensor(Exp_tar_adj.to_numpy())
        #print(set(genes_tar).issubset(genes))
        
        #Apply asin transform on expression data
        Exp_data_df = Exp_data_df.apply(np.arcsinh)
        #Generate paiwaise pearson coorelation of genes
        Exp_coe_data_np =  np.array(np.corrcoef(Exp_data_df.values))
        #Exp_coe_data_np[np.diag_indices_from(Exp_coe_data_np)] = 0
        Exp_coe_data_ten = torch.as_tensor(Exp_coe_data_np)
        #Calulate the average co expression of a gene aganst other genes
        Exp_coe_data_mean = Exp_coe_data_ten.mean(dim=1,keepdim=True)

        IsTf = torch.as_tensor(istf(genes))
        #is_tf2 = torch.as_tensor(genes.str.fullmatch('|,'.join(TFs.Aliases.values.tolist()),case=False),dtype=torch.bool)
        #tfor = is_tf or is_tf2
        #Generate Z scores for individual expression of a cell across genes
        exp_mat_forz = torch.as_tensor(Exp_data_df.T.to_numpy())
        
        Z_exp_acrossgenes = z_score(exp_mat_forz)
        minm_acrossgenes = min_max(Z_exp_acrossgenes)
        #print(minm_acrossgenes.shape)
        

        #norm_pest_intensity_tensor = (pest_intensity_tensor - pest_intensity_tensor.min()) / (pest_intensity_tensor.max() - pest_intensity_tensor.min())
        #generate Z scores of individual expression of a gene across cells
        Z_exp_acrosscell = z_score_dim1(exp_mat_forz)
        
        #exp_mat_df = torch.as_tensor(Exp_data_df.to_numpy())
        minmax_exp_acrosscell = min_max_dim0(Z_exp_acrosscell.T)
        print(minmax_exp_acrosscell.shape)
        
        return [minm_acrossgenes,IsTf,Exp_tar_adj_ten,Exp_coe_data_ten,Exp_coe_data_mean,minmax_exp_acrosscell,exp_mat_forz]#,gsummary]



def encode_paragraph(paragraph,model):
    sentences = paragraph.split(".")
    
    embeddings = model.encode(paragraph)
    #embeddings = np.mean(embeddings,axis=0,keepdims=True)
    return embeddings
def istf(genes):
    #Read human TFs list,and add a bool feature if a gene is a TF or not
    TFs = read_csv(DATASET_DIR+"/Data/TFHumans.csv",header=None)
    
    is_tf = genes.str.upper().isin(TFs[0])  
    #print(is_tf)
    return is_tf  
    #is_tf = genes.str.fullmatch('|'.join(TFs.Symbol.values.tolist()),case=False)
def clean_string(s):
    # Remove special characters and convert to lowercase
    s = re.sub(r'[^a-zA-Z0-9]', '', s)
    return s.lower()
def is_tf(genes):
    # Read human TFs list and add a bool feature if a gene is a TF or not
    # Replace with the actual path to your dataset
    TFs = read_csv(DATASET_DIR + "/Data/Human_genes.csv", delimiter="\t")
    
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
def min_max(exp_mat):
    max_n,_ = torch.max(exp_mat,dim=0)
    
    min_n,_ = torch.min(exp_mat,dim=0)
    exp_mat = (exp_mat - min_n ) / (max_n - min_n) 
    return exp_mat
def min_max_dim0(exp_mat):
    max_n,_ = torch.max(exp_mat,dim=0)
    
    min_n,_ = torch.min(exp_mat,dim=0)
    exp_mat = (exp_mat - min_n ) / (max_n - min_n) 
    return exp_mat.T
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

            edge_index = (abs(edge_weight_matrix) > 0.0).nonzero().t()
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
def activate_grn(exp_mat,exp_mean,is_tf):
       print(exp_mat.shape,exp_mean.shape,is_tf.shape)
       
       for i in range(exp_mat.size(0)):
        #    for j in range(0,exp_mat.size(1)):
                
                if ((exp_mat[i,j] < exp_mean[i]) and (is_tf[i]==1)):
                    exp_mat[i,j] = 0
                     
       return exp_mat
def filter_edges_by_nodes(edge_index, node_indices):
    mask = (edge_index[0] < len(node_indices)) & (edge_index[1] < len(node_indices))
    return edge_index[:, mask]
class GCENgraph(Dataset):
    def __init__(self, root, transform=None, target_transform=None,stage="train",num_sam = 1000):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        self.stage = stage
        self.num_sam = num_sam
        self.GCENset =[]
        self.SubaGraphs = []
        datas = GCENData(DATASET_DIR)
        
        for data in datas:
                
                Zscores_genes,is_tf,Target_grn,Coexp_mat,coexp_mean,zscores_cells,expr_mat = data
                #gsum_ten = torch.as_tensor(np.array(list(gsum.values())))
                #print((gsum_ten.shape))
                gcengraph = construct_networkx(Coexp_mat)
                mean = torch.mean(zscores_cells,axis=0,keepdim=True)
                
                
                for i in range(0,len(Zscores_genes)):
                        

                        
                        #node_att = torch.column_stack((is_tf,Zscores_genes[i],zscores_cells[i],coexp_mean))#,gsum_ten))
                        expr_cell = activate_grn(zscores_cells.t(),mean.t(),is_tf.t())
                        node_att = torch.column_stack((is_tf,expr_cell))
                        targ_grn_activated = Target_grn
                        
                        positive_edges = coo_tensor_create_pos(targ_grn_activated)
                        neg_edges = coo_tensor_create_neg(targ_grn_activated)
                        gcengraph.x = node_att
                        gcengraph.y = targ_grn_activated
                        gcengraph.edges = Coexp_mat
                        gcengraph.pos_edge_label_index = positive_edges
                        gcengraph.neg_edge_label_index = neg_edges
                        #gcengraph.gsum = gsum_ten
                        
                        self.GCENset.append(gcengraph)
                        #print((gcengraph))
                        values = list(np.arange(len(gcengraph.x)-2))
                        
                        #random.shuffle(values)
                        #covered_values=set()
                        for i in range(self.num_sam):
                            
                            #sample = torch.as_tensor(random.sample([value for value in values if value not in covered_values], 100))
                            sample = torch.as_tensor(random.sample(values, 100))
                            #print(sample)
                            # Add the sampled values to the set of covered values
                            
                            subg = gcengraph.subgraph(sample)
                            
                            subg.y = subg.y[:,sample]
                            subg.edges = subg.edges[:,sample]
                            is_tf_sam = is_tf[sample]
                            
                            #subg.gsum = gsum_ten[sample,:]
                            targ_grn_activated_sam = targ_grn_activated[:,sample][sample,:]
                            
                            positive_edges_sam = coo_tensor_create_pos(targ_grn_activated_sam)
                            neg_edges_sam = coo_tensor_create_neg(targ_grn_activated_sam)
                            if positive_edges_sam.numel()!=0:
                                subg.pos_edge_label_index=positive_edges_sam
                                #subg.neg_edge_label_index=neg_edges_sam
                                subg.neg_edge_label_index=negative_sampling(subg.pos_edge_label_index, subg.edges.size(0))
                                self.SubaGraphs.append(subg)
        
                            #targ_grn_activated = activate_grn(expr_mat[i],mean.T,Target_grn,is_tf.T)
                            
                            
                            '''
                            unique_nodes_pos = torch.unique(gcengraph.pos_edge_label_index.flatten())
                            
                            sample_indices_pos =torch.tensor([idx for idx in sample if idx in unique_nodes_pos])
                            if sample_indices_pos.numel() !=0:
                                
                                subg.pos_edge_label_index=subgraph(sample_indices_pos,gcengraph.pos_edge_label_index)[0]
                                
                            
                            unique_nodes_neg = torch.unique(gcengraph.neg_edge_label_index.flatten())
                            
                            sample_indices_neg =torch.tensor([idx for idx in sample if idx in unique_nodes_neg])
                            if sample_indices_neg.numel()!=0:
                                subg.neg_edge_label_index=subgraph(sample_indices_neg,gcengraph.neg_edge_label_index)[0]
                            '''                    
                            
        print(len(self.SubaGraphs))                    
                            
       
    def __len__(self):
        return len(self.SubaGraphs)
    
    
    def __getitem__(self, idx):
          data = self.SubaGraphs[idx].x
          #gsum = self.SubaGraphs[idx].gsum
          
          edges = self.SubaGraphs[idx].edge_index
          pos_edges = self.SubaGraphs[idx].pos_edge_label_index
          neg_edges = self.SubaGraphs[idx].neg_edge_label_index
          targ_mat = self.SubaGraphs[idx].y
          edge_attr = self.SubaGraphs[idx].edge_weight.float()
          co_epmat = self.SubaGraphs[idx].edges
          #item = next(GCENdataset)
          return [data,edges, pos_edges,neg_edges,targ_mat,edge_attr,co_epmat]#,gsum]   
          #item = self.GCENset[idx]
          #return data


dat = GCENgraph(DATASET_DIR)
print(dat[0][3].shape)
