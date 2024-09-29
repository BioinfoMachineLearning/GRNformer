"""
author: Akshata 
timestamp: Thu August 24th 2023 11.40 AM
"""
import torch_geometric as pyg
from torch import nn
#import graph_transformer_pytorch as gt


import math
import os
import glob
import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch.nn as nn
import numpy as np
from pandas import read_csv
from torch import  Tensor
import torch.nn.functional as F
import re
from torch.nn import Module
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.nn import GAE, VGAE, GCNConv,TransformerConv,Linear,BatchNorm, InnerProductDecoder
import wandb
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC,BinaryAveragePrecision,BinaryConfusionMatrix
from torch_geometric.utils import negative_sampling
import Dataset_activatinglabels_biobert_new as dt
from GRNFormerNewModels import GraphUNet
from GRNembedding import Embeddings
from argparse import ArgumentParser
from typing import Optional, Tuple
AVAIL_GPUS = [0,1]
NUM_NODES = 1
BATCH_SIZE = 1
DATALOADERS = 1
ACCELERATOR = "gpu"
EPOCHS = 200
NODES_DIM = 256
EXP_CHANNEL=426
BERT_CHANNEL=426
EDGE_DIM = 1
NUM_HEADS = 8
ENCODE_LAYERS =1 
OUT_CH=128
DATASET_DIR = "/home/aghktb/GRN/GRNformer"

EPS = 1e-15

"""

torch.set_default_tensor_type(torch.FloatTensor)  # Ensure that the default tensor type is FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose the device you want to use

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner to find the best algorithm to use for hardware
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # Set the default tensor type to CUDA FloatTensor
    torch.set_float32_matmul_precision('medium')  # Set Tensor Core precision to medium

"""

CHECKPOINT_PATH = f"{DATASET_DIR}/GRNUnet_newembedding_hHep_CSChipseq"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)


class GRNFormerLinkPred(pl.LightningModule):
    def __init__(self, learning_rate=1e-4,node_dim=NODES_DIM,out_chan=OUT_CH,num_heads=NUM_HEADS,edge_dim=EDGE_DIM,encoder_layers=ENCODE_LAYERS, **model_kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        #self.model = VGAE(VariationalGCNEncoder(node_dim, out_chan))
        self.embed = Embeddings(exp_channel=EXP_CHANNEL,bert_channel=BERT_CHANNEL,out_channels=NODES_DIM)
        self.model = GraphUNet(node_dim, hidden_channels=out_chan,out_channels=8,depth=encoder_layers)
        self.decoder = InnerProductDecoder()
        self.criterian = nn.BCEWithLogitsLoss()
        self.loss_fn = self.recon_loss
        self.kl =self.model.kl_loss
        self.metrics = MetricCollection([BinaryAUROC(),BinaryAveragePrecision()])
        #self.metrics = self.test
        #self.train_metrics = self.metrics.clone(prefix="train_")
        self.train_metrics = self.test
        #self.valid_metrics = self.metrics.clone(prefix="valid_")
        self.valid_metrics = self.test
        #self.test_metrics = self.metrics.clone(prefix="test_")
        self.test_metrics = self.test

    def forward(self, exp_data,bert_data,edge_sampled,edge_attr):

        x = self.embed(exp_data,bert_data)
        x = x.squeeze()
        grnedges_lat = self.model(x,edge_sampled,edge_attr)
        #grnedges = self.model.decode(grnedges_lat)
        return grnedges_lat
    
    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Tensor) -> Tensor:
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index) +
                              EPS).mean()

        return pos_loss + neg_loss
    
    def test(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
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
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index)
        neg_pred = self.decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        preds = pred
        return roc_auc_score(y, preds), average_precision_score(y, preds)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, eps=1e-10, verbose=True)
        metric_to_track = 'valid_loss'
        return{'optimizer':optimizer,
               'lr_scheduler':lr_scheduler,
               'monitor':metric_to_track}
    
    def training_step(self,batch,batch_idx):
        batch_exp_data = batch[0].cuda()
        batch_bert_data = batch[0].cuda()
        
        batch_edges = batch[1].squeeze(0).cuda()
        batch_edge_attr =torch.transpose(batch[5],0,1).float().cuda()


        grn_pred_edge = self.forward(batch_exp_data,batch_bert_data,batch_edges,batch_edge_attr)
        batch_targ_pos = batch[2].squeeze(0).cuda()       
        batch_targ_neg = batch[3].squeeze(0).cuda()
        batch_tar_mat = batch[4].float().cuda()
        
        #loss = (self.loss_fn(grn_pred_edge.float(),batch_tar_mat.view(-1))+EPS).mean()
        loss = self.loss_fn(grn_pred_edge,batch_targ_pos,batch_targ_neg)
        #loss = loss_pos.mean()+loss_neg.mean()
        loss = loss + (1 / len(batch_exp_data)) #* self.kl()
        
        metric_auroc,metric_ap = self.train_metrics(grn_pred_edge,batch_targ_pos,batch_targ_neg)
        self.log_dict({"train_auroc":metric_auroc,"train_ap":metric_ap},sync_dist=True)
   
        
        self.log('train_loss',loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        batch_exp_data = batch[0].cuda()
        batch_bert_data = batch[0].cuda()
        
        batch_edges = batch[1].squeeze(0).cuda()
        batch_edge_attr =torch.transpose(batch[5],0,1).float().cuda()


        grn_pred_edge = self.forward(batch_exp_data,batch_bert_data,batch_edges,batch_edge_attr)
        batch_targ_pos = batch[2].squeeze(0).cuda()       
        batch_targ_neg = batch[3].squeeze(0).cuda()
        batch_tar_mat = batch[4].float().cuda()
        
        #loss = (self.loss_fn(grn_pred_edge.float(),batch_tar_mat.view(-1))+EPS).mean()
        loss = self.loss_fn(grn_pred_edge,batch_targ_pos,batch_targ_neg)
        #loss = loss_pos.mean()+loss_neg.mean()
        loss = loss + (1 / len(batch_exp_data.squeeze())) #* self.kl()
        #loss = loss + (1 / len(batch_data)) * self.kl()
        metric_auroc,metric_ap = self.valid_metrics(grn_pred_edge,batch_targ_pos,batch_targ_neg)
        self.log_dict({"valid_auroc":metric_auroc,"valid_ap":metric_ap},sync_dist=True)
        self.log('valid_loss',loss, on_step=True, on_epoch=True, sync_dist=True)
       
    
    def test_step(self,batch, batch_idx):
        batch_exp_data = batch[0].cuda()
        batch_bert_data = batch[0].cuda()
        
        batch_edges = batch[1].squeeze(0).cuda()
        batch_edge_attr =torch.transpose(batch[5],0,1).float().cuda()


        grn_pred_edge = self.forward(batch_exp_data,batch_bert_data,batch_edges,batch_edge_attr)
        batch_targ_pos = batch[2].squeeze(0).cuda()       
        batch_targ_neg = batch[3].squeeze(0).cuda()
        batch_tar_mat = batch[4].float().cuda()
        
        #loss = (self.loss_fn(grn_pred_edge.float(),batch_tar_mat.view(-1))+EPS).mean()
        loss = self.loss_fn(grn_pred_edge,batch_targ_pos,batch_targ_neg)
        #loss = loss_pos.mean()+loss_neg.mean()
        loss = loss + (1 / len(batch_exp_data)) #* self.kl()
        #loss = loss + (1 / len(batch_data)) * self.kl()
        metric_auroc,metric_ap = self.test_metrics(grn_pred_edge,batch_targ_pos,batch_targ_neg)
        self.log_dict({"test_auroc":metric_auroc,"test_ap":metric_ap},sync_dist=True)
        
       
        #conf_mat = BinaryConfusionMatrix().to("cuda")
        #conf_vals = conf_mat(grn_pred_edge,batch_tar_mat.view(-1).int())
        #print("Test Data Confusion Matrix: \n")
        #print(conf_vals)
        
        self.log('test_loss',loss, on_step=True, on_epoch=True, sync_dist=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--node_dim',type=int,default=NODES_DIM)
        parser.add_argument('--encoder_layers',type=int,default=ENCODE_LAYERS)
        parser.add_argument('--edge_dim',type=int,default=EDGE_DIM)
        return parser

def train_GRNFormerLinkPred():
    pl.seed_everything(123)
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GRNFormerLinkPred.add_model_specific_args(parser)
    parser.add_argument('--num_gpus', type=int, default=AVAIL_GPUS,
                        help="Number of GPUs to use (e.g. -1 = all available GPUs)")
    parser.add_argument('--nodes', type=int, default=NUM_NODES, help="Number of nodes to use")
    parser.add_argument('--num_epochs', type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int,
                        help="effective_batch_size = batch_size * num_gpus * num_nodes")
    parser.add_argument('--num_dataloader_workers', type=int, default=DATALOADERS)
    parser.add_argument('--entity_name', type=str, default='aghktb', help="Weights and Biases entity name")
    parser.add_argument('--project_name', type=str, default='GRNFormerLinkPred',
                        help="Weights and Biases project name")
    
    parser.add_argument('--save_dir', type=str, default=CHECKPOINT_PATH, help="Directory in which to save models")
    
    parser.add_argument('--unit_test', type=int, default=False,
                        help="helps in debug, this touches all the parts of code."
                             "Enter True or num of batch you want to send, " "eg. 1 or 7")
    args = parser.parse_args()
    
    args.devices = args.num_gpus
    args.num_nodes = args.nodes
    args.accelerator = ACCELERATOR
    args.max_epochs = args.num_epochs
    args.fast_dev_run = args.unit_test
    args.log_every_n_steps = 1
    args.detect_anomaly = True
    args.enable_model_summary = True
    args.weights_summary = "full"
    
    dataset = dt.GCENgraph(DATASET_DIR)
  
    print(len(dataset))
    train_size = int(0.71 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - (train_size+val_size)
    dataset_train,dataset_valid,dataset_test = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])
    
    #dataset_valid = MicrographDataValid(DATASET_DIR)
    #dataset_test = MicrographDataValid(DATASET_DIR) # using validation data for testing here
    train_loader = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_dataloader_workers)
    print(train_size)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_dataloader_workers)
    test_loader = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_dataloader_workers)
    #torch.save(test_loader,DATASET_DIR+'/test.pt')
    model = GRNFormerLinkPred(learning_rate=1e-4)
    
    trainer = pl.Trainer.from_argparse_args(args)
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss', save_top_k=10, dirpath=args.save_dir, filename='GRNFormer_beeline_{epoch:02d}_{valid_loss:6f}')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopping_callback = EarlyStopping(monitor='valid_loss', mode='min', min_delta=0.0, patience=20)
    trainer.callbacks = [checkpoint_callback, lr_monitor, early_stopping_callback]
    logger = WandbLogger(project=args.project_name, entity=args.entity_name, name=args.save_dir, offline=False, save_dir=".")
    trainer.logger = logger
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(dataloaders=test_loader, ckpt_path='best')
   



if __name__ == "__main__":
    train_GRNFormerLinkPred()
