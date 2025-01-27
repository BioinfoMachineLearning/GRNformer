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
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch.nn as nn
import numpy as np
from pandas import read_csv
from torch import  Tensor
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,ConcatDataset

import wandb
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC,BinaryAveragePrecision,BinaryConfusionMatrix,BinaryF1Score,BinaryAccuracy,BinaryJaccardIndex,BinaryPrecision,BinaryRecall
from torch_geometric.utils import negative_sampling
import DatasetMaker.DatasetwithTFcenter_inference_grid as dt
from GRNFormerNewModels import EdgeTransformerEncoder_tcn,TransformerDecoder_tcn,EdgePredictor,VGAE,Reconstruct
from Scripts.GeneTranscoder.NewEMbedderModel import TransformerAutoencoder
from argparse import ArgumentParser
from torch_geometric.utils import to_dense_adj
from typing import Optional, Tuple
from torch_geometric.loader import DataListLoader
from sklearn.decomposition import PCA
from dotenv import load_dotenv

load_dotenv()
AVAIL_GPUS = [0]
NUM_NODES = 1
BATCH_SIZE = 1
DATALOADERS = 1
ACCELERATOR = "gpu"
EPOCHS = 100
NODES_DIM = 64
EXP_CHANNEL=426
BERT_CHANNEL=426
EDGE_DIM = 1
NUM_HEADS = 4
ENCODE_LAYERS = 3
OUT_CH=16
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


CHECKPOINT_PATH = f"{DATASET_DIR}/GRNformer_TFcenter_ChipseqhHep_6l_4h_16o"


class GRNFormerLinkPred(pl.LightningModule):
    def __init__(self,learning_rate=1e-4,node_dim=NODES_DIM,out_chan=OUT_CH,num_heads=NUM_HEADS,edge_dim=EDGE_DIM,encoder_layers=ENCODE_LAYERS, **model_kwargs):
        super().__init__()
        
        self.save_hyperparameters()

        self.embed = TransformerAutoencoder(embed_dim=NODES_DIM,nhead=4,num_layers=1)
        self.model = VGAE(encoder=EdgeTransformerEncoder_tcn(node_dim, out_chan,num_head=num_heads,edge_dim=edge_dim,num_layers=encoder_layers),decoder=TransformerDecoder_tcn(latent_dim=out_chan,out_channels=out_chan,num_head=num_heads,num_layers=encoder_layers,edge_dim=edge_dim))
        self.edgepred = EdgePredictor(latent_dim=out_chan)
        self.reconstruct = Reconstruct()
        self.criterian = nn.BCELoss()
        self.loss_fn = self.recon_loss
        self.kl =self.model.kl_loss
        self.metrics = MetricCollection([BinaryAUROC(),BinaryAveragePrecision(),BinaryF1Score(threshold=0.5),BinaryAccuracy(threshold=0.5),BinaryJaccardIndex(threshold=0.5),BinaryRecall(threshold=0.5),BinaryPrecision(threshold=0.5)])
        self.test_metrics_1 = self.metrics.clone(prefix="test_")
        self.test_metrics = self.test

    def forward(self, exp_data,bert_data,edge_sampled,edge_attr):

        z = self.embed(exp_data)
        #print(z.shape)
        x = z.squeeze()
        grnnodes_lat, grnedges_lat,mu,logstd= self.model.encode(x,edge_sampled,edge_attr)
        grnnodes_lat,grn_edges,grn_edge_att = self.model.decode(grnnodes_lat,grnedges_lat[0],grnedges_lat[1])
        #print(grnedges_lat[0],grn_edges)
        edge_prob =self.edgepred(grnnodes_lat,grn_edges,grn_edge_att,sigmoid=True)
        #print(edge_prob.shape)
        return edge_prob,z,mu,logstd
    
    def recon_loss(self, z: Tensor,emb:Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor]=None) -> Tensor:
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        
        #print(pos_edge_index)
        """
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0),num_neg_samples=pos_edge_index.size(1))
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)
        #print(z.shape,pos_edge_index)
        pos_pred = self.reconstruct(z, pos_edge_index,sigmoid=True)
        neg_pred = self.reconstruct(z, neg_edge_index,sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
    
        #pred = z.view(-1)
        #num_nodes = z.size(0)
        #y = to_dense_adj(pos_edge_index, max_num_nodes=num_nodes).squeeze(0)
        # Flatten the adjacency matrix and predicted edge probabilities
        #y_flat = y.view(-1)
        ##print(pred.shape,y.shape,y_flat.shape)
        loss = self.criterian(pred,y)
        #klloss = self.kl()
        return loss,neg_edge_index
    
    def test(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Optional[Tensor]=None,prefix=None) -> Tuple[Tensor, Tensor]:
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0),num_neg_samples=pos_edge_index.size(1))
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.reconstruct(z, pos_edge_index,sigmoid=True)
        neg_pred = self.reconstruct(z, neg_edge_index,sigmoid=True)
        preds = torch.cat([pos_pred, neg_pred], dim=0)
        #print(pos_pred,neg_pred)
        #y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        #preds = z.view(-1)
        #num_nodes = z.size(0)
        #y = to_dense_adj(pos_edge_index, max_num_nodes=num_nodes).squeeze(0)
        # Flatten the adjacency matrix and predicted edge probabilities
        y_flat = y
        if prefix=="train":
            metrics = self.train_metrics_1(preds,y_flat.int())
            return metrics
        elif prefix=="valid":
            metrics = self.valid_metrics_1(preds,y_flat.int())
            return metrics
        elif prefix=="test":
            metrics = self.test_metrics_1(preds,y_flat.int())
            conf_mat = BinaryConfusionMatrix(threshold=0.5)
            conf_vals = conf_mat(preds.detach().cpu(),y.detach().cpu())
            return metrics,conf_vals,pos_pred
        else:
            metrics = self.metrics(preds,y_flat.int())
            return metrics

    def test_step(self,batch, batch_idx):

        batch_exp_data = batch[0][0].unsqueeze(0).float().cuda()
        
        batch_bert_data = batch[0][0].cuda()
        batch_edges = batch[0][1].squeeze(0).cuda()
        batch_edge_attr = torch.transpose(batch[0][2].unsqueeze(0),0,1).float().cuda()


        grn_pred_edge,z,mu,logstd  = self.forward(batch_exp_data,batch_bert_data,batch_edges,batch_edge_attr)
        batch_targ_pos = batch[0][3].cuda()  
        
        loss,neg_edge = self.loss_fn(grn_pred_edge,z,batch_targ_pos)
        
        klloss = self.kl(mu,logstd)*1/len(z)
        loss=loss+klloss
        metrics,conf_val,preds = self.test_metrics(grn_pred_edge,batch_targ_pos,neg_edge_index=neg_edge,prefix="test")    
        # self.log_dict(metrics,sync_dist=True)
        # fig, ax = plt.subplots()
        # sns.heatmap(conf_val , annot=True, cmap="Blues", fmt="d")
        # plt.title("Confusion Matrix")
        # plt.tight_layout()
        # wandb.log({f"Confusion Matrix" :wandb.Image(fig)})
        # self.log('test_loss',loss, on_step=True, on_epoch=True, sync_dist=True)
        return {f'preds_grn' : grn_pred_edge,f'target_grnindex':batch_targ_pos,f'preds':preds,f'nodemap':batch[0][4]}
    
    def add_index_and_column(self,exp_file, final_predictions):
        # Read the expression file
        expression_data = pd.read_csv(exp_file,index_col=0)
        
        # Get the gene names from the index of the expression file
        gene_names = expression_data.index.tolist()
        
        # Add the gene names as the column names of final_predictions
        final_predictions.columns = gene_names
        
        # Add the gene names as the index of final_predictions
        final_predictions.index = gene_names
        
        return final_predictions, gene_names
    def test_epoch_end(self,outputs):
        # Log individual results for each dataset
        print(len(outputs))
        num_nodes = self.hparams.totalnodes
        link_prediction_matrix = torch.zeros((num_nodes, num_nodes)).cuda()
        counts_matrix = torch.zeros((num_nodes, num_nodes)).cuda()
        all_edges=[]
        if args.TFspecies=="human":
            tf_genes  = pd.read_csv(os.path.abspath("Data/sc-RNA-seq/hESC/TFHumans.csv"),header=None)[0].to_list()
        elif args.TFspecies=="mouse":
            tf_genes = pd.read_csv(os.path.abspath("Data/sc-RNA-seq/mESC/TFMouse.csv"))['TF'].to_list()
        tf_weight = 2  # Weight for TF genes
        # Assuming `outputs` is a list of dictionaries, and link_prediction_matrix and counts_matrix are predefined tensors
        expression_data = pd.read_csv(args.expFile,index_col=0)
        
        # Get the gene names from the index of the expression file
        gene_names = expression_data.index.tolist()
        for sam in range(len(outputs)):
            # Extract current sample data
            dataset_outputs = outputs[sam]['preds_grn']
            target_grn = outputs[sam]['target_grnindex']
            node_mappings = outputs[sam]['nodemap']
            
            # Generate reverse mapping in tensor form for faster indexing
            max_index = max(node_mappings.values())  # Ensure the tensor size fits all indices
            reverse_mapping = torch.zeros(max_index + 1, dtype=torch.long, device=dataset_outputs.device)
            for orig_idx, subgraph_idx in node_mappings.items():
                reverse_mapping[subgraph_idx] = orig_idx
        
            # Convert `dataset_outputs` indices to original graph indices using `reverse_mapping`
            original_indices = reverse_mapping[torch.arange(dataset_outputs.size(0)).to(dataset_outputs.device)]
            #print(original_indices)
            # Create batch indices for all (i, j) pairs in dataset_outputs
            batch_indices = torch.cartesian_prod(original_indices, original_indices).T
            #print(batch_indices)
            batch_probs = dataset_outputs.flatten()
            # #Create a mask for probabilities greater than 0.5
            mask = batch_probs > 0.5

            tf_indices = [index for index, gene in enumerate(gene_names) if gene in tf_genes]  # Assuming `tf_genes` is a list of TF gene names
            non_tf_indices = [index for index, gene in enumerate(gene_names) if gene not in tf_genes]
            # Convert tf_indices and non_tf_indices to tensors for compatibility with indexing
            tf_indices = torch.tensor(tf_indices, dtype=torch.long).cuda()
            # link_prediction_matrix.index_put_((batch_indices[0][mask][tf_indices], batch_indices[1][mask][tf_indices]),  # Filtered indices of TF genes based on mask
            #                                   batch_probs[mask][tf_indices] * (counts_matrix[batch_indices[0], batch_indices[1]][mask][tf_indices] + 1) * tf_weight,  # Filtered probabilities of TF genes based on mask
            #                                   accumulate=True
            #                                   )
            
            # link_prediction_matrix.index_put_((batch_indices[0][mask][non_tf_indices], batch_indices[1][mask][non_tf_indices]),  # Filtered indices of non-TF genes based on mask
            #                                   batch_probs[mask][non_tf_indices] * (counts_matrix[batch_indices[0], batch_indices[1]][mask][non_tf_indices] + 1),  # Filtered probabilities of non-TF genes based on mask
            #                                   accumulate=True
            #                                   )
            # counts_matrix.index_put_((batch_indices[0][mask], batch_indices[1][mask]), 
            #                          torch.ones_like(batch_probs[mask]), 
            #                          accumulate=True)
            
            # Separate masked batch indices and probabilities
            masked_batch_indices_0 = batch_indices[0][mask]
            masked_batch_indices_1 = batch_indices[1][mask]
            masked_batch_probs = batch_probs[mask]

            # Determine if masked_batch_indices contain TF genes
            is_source_tf = torch.isin(masked_batch_indices_0, tf_indices)
            is_target_tf = torch.isin(masked_batch_indices_1, tf_indices)
            is_tf_edge = is_source_tf  # True if either source or target is TF gene

            # Create weights based on TF presence
            tf_weights = torch.where(is_tf_edge, tf_weight, 1.0)  # Apply tf_weight where either node is a TF

            # Calculate weighted probabilities
            weights = masked_batch_probs * (counts_matrix[masked_batch_indices_0, masked_batch_indices_1] + 1) * tf_weights

            # Accumulate the weighted predictions in the link prediction matrix
            link_prediction_matrix.index_put_(
                (masked_batch_indices_0, masked_batch_indices_1),
                weights,
                accumulate=True
            )

            # Update the counts matrix
            counts_matrix.index_put_(
                (masked_batch_indices_0, masked_batch_indices_1),
                torch.ones_like(masked_batch_probs),
                accumulate=True
            )
            # Map target_grn edges back to the original graph indices using reverse_mapping
            edge_index_sampled = target_grn
            old_edge_indices = reverse_mapping[edge_index_sampled].cuda()
        
            # Append edges directly to all_edges without sorting here
            all_edges.append(old_edge_indices) 
       
        all_edges1 = torch.cat(all_edges, dim=1)
        # Use torch.unique to remove duplicates and get unique edges
        #unique_edges =  torch.unique(all_edges, dim=1)
        unique_edges = set(map(tuple, all_edges1.t().tolist()))
        print(len(unique_edges))
        # Convert the set of tuples to a list of lists
        unique_edges_list = list(unique_edges)

        
        # Convert to a tensor and transpose to match edge_index format
        edge_index_unique = torch.tensor(unique_edges_list).t()
        print(edge_index_unique.shape)
        # Avoid division by zero by adding a small epsilon
        counts_matrix[counts_matrix == 0] = 1e-8
        #print(link_prediction_matrix,counts_matrix)
        # Compute final predictions using weighted average
        link_prediction_matrix=link_prediction_matrix.fill_diagonal_(0)
        final_predictions =torch.sigmoid(link_prediction_matrix / counts_matrix)
        final_predictions[final_predictions <= 0.5] = 0
        
        #final_predictions1, indices = torch.sort(final_predictions)
        df = pd.DataFrame(final_predictions.cpu().numpy())
        
        metrics,conf_val,preds = self.test_metrics(final_predictions,edge_index_unique,prefix="test")
        print(metrics)
        # Log final predictions (or return if needed)
        final_predictions1,gene_names = self.add_index_and_column(args.expFile, df)
        print(final_predictions1)
        
        # Filter predictions with probability greater than 0.5
        filtered_predictions = final_predictions1[final_predictions1 > 0.9].stack().reset_index()
        filtered_predictions.columns = ['gene1', 'gene2', 'probability']
        sorted_df = filtered_predictions.sort_values(by='probability', ascending=False)
        #top_k_rows = sorted_df.head(len(unique_edges)*2)
        top_k_rows = sorted_df[sorted_df['gene1'].isin(tf_genes)]
        
        # Save filtered predictions to CSV file
        top_k_rows.to_csv(args.outPrefix+"/final_predictions0_8.csv", index=False)
        #print(top_k_rows)
        # Assuming you have a list of gene names called `gene_names`
        gene_names_dict = {i: gene_names[i] for i in range(len(gene_names))}

        # Convert the unique edges list to include gene names
        unique_edges_with_names = [(gene_names_dict[edge[0]], gene_names_dict[edge[1]]) for edge in unique_edges_list]
        #print(unique_edges_with_names)
        # Assuming you have the lists top_k_rows and unique_edges_with_names

        # Convert top_k_rows to a list of tuples
        top_k_edges = [(row['gene1'], row['gene2']) for _, row in top_k_rows.iterrows()]
        #print(top_k_edges)
        # Count the number of matches between top_k_edges and unique_edges_with_names
        num_matches = sum(edge in unique_edges_with_names for edge in top_k_edges)

        print("Number of matches:", num_matches)
        #self.log_dict(metrics,sync_dist=True)
        #fig, ax = plt.subplots()
        #sns.heatmap(conf_val , annot=True, cmap="Blues", fmt="d")
        #plt.title("Confusion Matrix")
        #plt.tight_layout()
        #wandb.log({f"Confusion Matrix" :wandb.Image(fig)})
        
        return final_predictions
            

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--node_dim',type=int,default=NODES_DIM)
        parser.add_argument('--encoder_layers',type=int,default=ENCODE_LAYERS)
        parser.add_argument('--edge_dim',type=int,default=EDGE_DIM)
        parser.add_argument('--totalnodes', type=int,default=500)
        parser.add_argument('--tf_list', type=torch.tensor,default=torch.tensor([1,2,3,4,5,6,7,8,9,10]))
        return parser


if __name__ == "__main__":
    pl.seed_everything(123)
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GRNFormerLinkPred.add_model_specific_args(parser)
    
    parser.add_argument('--expFile',type=str, default=False,
                        help="sets the expression file of datafolder"
                             "Enter the relative path to the root folder of the dataset eg:'Data/sc-RNA-seq/hESC/hESC_nonspecific_chipseq_500-ExpressionData.csv'" )
    parser.add_argument('--netFile',type=str, default=False,
                        help="sets the ground truth network of datafolder"
                             "Enter the relative path to the root folder of the dataset eg:'Data/sc-RNA-seq/hESC/hESC_nonspecific_chipseq_500-network1.csv'" )
    parser.add_argument('--TFspecies',type=str, default=False,
                        help="sets the TF file of datafolder"
                             "Enter the relative path to the root folder of the dataset eg:'human/mouse'" )
    parser.add_argument('--outPrefix',type=str, default=False,
                        help="sets the expression file of datafolder"
                             "Enter the relative path to the root folder of the dataset eg:'Data/sc-RNA-seq/hESC/hESC_nonspecific_chipseq_500-ExpressionData.csv'" )
    args = parser.parse_args()
    
    args.devices = -1
    args.num_nodes = 1
    args.accelerator = ACCELERATOR
    args.detect_anomaly = True
    args.enable_model_summary = True
    args.weights_summary = "full"
    # root=[os.path.abspath(args.root)]
    root = [os.path.dirname(os.path.abspath(args.expFile))]
    gene_expression_file=[os.path.abspath(args.expFile)]
    numnodes= len(pd.read_csv(os.path.abspath(args.expFile)))
    if args.TFspecies=="human":
        tf  = pd.read_csv(os.path.abspath("Data/sc-RNA-seq/hESC/TFHumans.csv"),header=None)[0].to_list()
    elif args.TFspecies=="mouse":
        tf = pd.read_csv(os.path.abspath("Data/sc-RNA-seq/mESC/TFMouse.csv"))['TF'].to_list()
    else:
        print("species not supported")
    TF_list = [tf]
    regulation_file=[os.path.abspath(args.netFile)]
    #os.makedirs(DATASET_DIR+"/"+args.save_dir, exist_ok=True)
    All_test_dataset=[]
    for i in range(len(root)):

        dataset = dt.GeneExpressionDataset(root[i],gene_expression_file[i],TF_list[i],regulation_file[i])

    All_test_dataset.append(dataset)

    TestDatasets = ConcatDataset(All_test_dataset)
  
    test_loader = DataListLoader(dataset=TestDatasets, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    model = GRNFormerLinkPred(learning_rate=1e-3,totalnodes=numnodes)
    print("Model loaded")
    trainer = pl.Trainer.from_argparse_args(args)
    #logger = WandbLogger(project=args.project_name, entity=args.entity_name, name=args.save_dir, offline=False, save_dir=".")
    #trainer.logger = logger
    #trainer.fit(model, train_loader, valid_loader,ckpt_path=os.path.abspath("/GRNFormer_TFCENTER_NegEdges_withdecode_v2_resume/GRNFormer_beeline_epoch=52_valid_loss=0.587835.ckpt")
    trainer.test(model,dataloaders=test_loader, ckpt_path=os.path.abspath('Trainings/tempo_leakyRelu_workplease/GRNFormer_beeline_epoch=26_valid_loss=0.645546.ckpt'))

