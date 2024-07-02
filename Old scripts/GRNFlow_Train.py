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
import sys
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch.nn as nn
import numpy as np
from pandas import read_csv
from torch import  Tensor
import torch.nn.functional as F
import re
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.nn import GAE, VGAE, GCNConv,TransformerConv,Linear,BatchNorm,GATConv
import wandb
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score,BinaryAUROC,BinaryConfusionMatrix,BinaryAveragePrecision,MultilabelAUROC,MultilabelAveragePrecision
from torch.nn import BCELoss,BCEWithLogitsLoss
import Dataset_activatinglabels_biobert as dt
import Dataset_activatinglabels_biobert_test as dt_test
from GraphFlow_Model import Encoder,Decoder,GNF
from argparse import ArgumentParser
from torch_geometric.loader import NeighborLoader
from torch.distributions import Normal
from torch_geometric.utils import negative_sampling
from torch_geometric.utils.convert import from_scipy_sparse_matrix
AVAIL_GPUS = [0]
NUM_NODES = 1
BATCH_SIZE = 1
DATALOADERS = 1
ACCELERATOR = "gpu"
EPOCHS = 40
NODES_DIM = 4
EDGE_DIM = 1
NUM_HEADS = 8
ENCODE_LAYERS = 6
OUT_CH=4
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

CHECKPOINT_PATH = "GRNFormer_tempo"
#os.makedirs(CHECKPOINT_PATH, exist_ok=True)


    


class GRNFormerLinkPred(pl.LightningModule):
    def __init__(self, learning_rate=1e-4,node_dim=NODES_DIM,out_chan=OUT_CH,num_heads=NUM_HEADS,edge_dim=EDGE_DIM,encoder_layers=ENCODE_LAYERS, **model_kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        #self.model = VGAE(VariationalGCNEncoder(node_dim, out_chan))
        #self.model = VGAE(EdgeTransformerEncoder_new(node_dim, out_chan,num_head=num_heads,edge_dim=edge_dim,num_layers=encoder_layers))
        self.encoder = Encoder(in_channels=node_dim,out_channels=out_chan,heads=num_heads)
        self.decoder = Decoder(in_channels=node_dim,out_channels=out_chan,heads=num_heads)
        self.gnf = GNF()
        self.loss_fn = self.recon_loss
        #self.loss_fn = BCELoss()
        #self.kl =self.model.kl_loss
        self.metrics = MetricCollection([BinaryAUROC(),BinaryAveragePrecision(),BinaryF1Score()])
        #self.metrics = self.model.test
        self.train_metrics = self.metrics.clone(prefix="train_")
        #self.train_metrics = self.model.test
        self.valid_metrics = self.metrics.clone(prefix="valid_")
        #self.valid_metrics = self.model.test
        self.test_metrics = self.metrics.clone(prefix="test_")
        #self.test_metrics = self.model.test

    def forward(self, data_sampled,edge_sampled,edge_attr):
        #grnedges_lat = self.model.encode(data_sampled,edge_sampled,edge_attr)
        #grnedges = self.model.decode(grnedges_lat)
        data_, edge_index,edge_attr1 = self.encoder(data_sampled,edge_sampled,edge_attr)
        # Forward

        z1,z2,log_det_xz = self.gnf.forward(data_,edge_index)

        log_det_xz = log_det_xz.unsqueeze(0)
        log_det_xz = log_det_xz.mm(torch.ones(log_det_xz.t().size()).cuda())
        #self.log_det_xz.append(log_det_xz)



        # Sample
        z = torch.cat((z1,z2),dim=1)
        print(z)
        n = Normal(z.cuda(), torch.ones(z.size()).cuda())
        z = n.sample()
        print("samples:",z)
        # Inverse
        z1,z2, log_det_zx = self.gnf.inverse(z,edge_index)
        log_det_zx = log_det_zx.unsqueeze(0).cuda()
        log_det_zx = log_det_zx.mm(torch.ones(log_det_zx.t().size()).cuda())
        print(log_det_zx)
        # Decoder
        z = torch.cat((z1,z2),dim=1)
        print(z)
        y, edge_ind = self.decoder(z,edge_index,edge_attr1)
        #y_ind,_ = from_scipy_sparse_matrix(y)
        return y,edge_ind
    
    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Tensor, target_adj: Tensor) -> Tensor:
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
        pos_edges = target_adj
        print(pos_edges.view(-1),z.view(-1))
        pos_loss = torch.nn.functional.binary_cross_entropy(z.view(-1),pos_edges.view(-1).float(),reduction="mean")

        neg_edges = 1- target_adj
        neg_loss = torch.nn.functional.binary_cross_entropy((1-z).view(-1),neg_edges.view(-1).float(),reduction="mean")
        print(pos_loss+neg_loss)
        return pos_loss + neg_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, eps=1e-10, verbose=True)
        metric_to_track = 'valid_loss'
        return{'optimizer':optimizer,
               'lr_scheduler':lr_scheduler,
               'monitor':metric_to_track}
    
    def training_step(self,batch,batch_idx):
        batch_data = batch[0].squeeze(0).cuda()
        batch_edges = batch[1].squeeze(0).cuda()
        batch_edge_attr = torch.transpose(batch[5],0,1).float().cuda()
        bath_targ = batch[4].squeeze(0).cuda()
        
        
        grn_pred_edge,pred_edgeindex = self.forward(batch_data,batch_edges,batch_edge_attr)
        batch_targ_pos = batch[2].squeeze(0).cuda()       
        batch_targ_neg = batch[3].squeeze(0).cuda()
        print(grn_pred_edge)
        
        loss = self.loss_fn(grn_pred_edge,batch_targ_pos,batch_targ_neg,bath_targ)
        #loss = loss + ( self.kl()/ len(batch_data)) 
        #all_edge_attr = torch.cat([batch_targ_pos,batch_targ_neg],dim=1)
        
        
        metric_train = self.train_metrics(grn_pred_edge.view(-1).cpu(),bath_targ.view(-1).cpu())
        #self.log_dict({"train_auroc":metric_auroc,"train_ap":metric_ap})
        self.log_dict(metric_train)
        
        self.log('train_loss',loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        batch_data = batch[0].squeeze(0).cuda()
        
        batch_edges = batch[1].squeeze(0).cuda()
        batch_edge_attr =torch.transpose(batch[5],0,1).float().cuda()
        grn_pred_edge,latz = self.forward(batch_data,batch_edges,batch_edge_attr)
        batch_targ_pos = batch[2].squeeze(0).cuda()       
        batch_targ_neg = batch[3].squeeze(0).cuda()
        bath_targ = batch[4].squeeze(0).cuda()
        loss = self.loss_fn(grn_pred_edge,batch_targ_pos,batch_targ_neg,bath_targ)
         
        #loss = loss + (1 / len(batch_data)) * self.kl()
        #all_edge_attr = torch.cat([batch_targ_pos,batch_targ_neg],dim=1)
        #print(grn_pred_edge.view(-1),bath_targ.view(-1))
        
        metric_valid = self.valid_metrics(grn_pred_edge.view(-1).cpu(),bath_targ.view(-1).cpu())

        #self.log_dict({"valid_auroc":metric_auroc,"valid_ap":metric_ap})
        self.log_dict(metric_valid)
        self.log('valid_loss',loss, on_step=True, on_epoch=True, sync_dist=True)
        
    
    def test_step(self,batch, batch_idx,dataloader_idx):
        batch_data = batch[0].squeeze(0).cuda()
        batch_edges = batch[1].squeeze(0).cuda()
        batch_edge_attr =torch.transpose(batch[5],0,1).float().cuda()
        grn_pred_edge,latz = self.forward(batch_data,batch_edges,batch_edge_attr)
        batch_targ_pos = batch[2].squeeze(0).cuda()
        print(batch_targ_pos.shape)       
        batch_targ_neg = batch[3].squeeze(0).cuda()
        bath_targ = batch[4].squeeze(0).cuda()
        loss = self.loss_fn(grn_pred_edge,batch_targ_pos,batch_targ_neg,bath_targ)
        
        #loss = loss + (1 / len(batch_data)) * self.kl()
        #all_edge_attr = torch.cat([batch_targ_pos,batch_targ_neg],dim=1)
        metric_test = self.test_metrics(grn_pred_edge.view(-1),bath_targ.view(-1))
        #self.log_dict({"test_auroc":metric_auroc,"test_ap":metric_ap})
        self.log_dict(metric_test)
       
        #conf_mat = MulticlassConfusionMatrix(num_classes=self.hparams.n_class).to("cuda")
        #conf_vals = conf_mat(class_pred, batch_label_class.squeeze())
        #print("Test Data Confusion Matrix: \n")
        #print(conf_vals)
        
        self.log('test_loss',loss, on_step=True, on_epoch=True, sync_dist=True)

    def test_epoch_end(self, outputs):
        # Log individual results for each dataset
        
        for i  in range(len(outputs)):
            dataset_outputs = outputs[i]
            class_preds = torch.cat([x[f'preds_class{i}'] for x in dataset_outputs])
            class_targets = torch.cat([x[f'targets_class{i}'] for x in dataset_outputs])
            conf_mat = BinaryConfusionMatrix(num_classes=self.hparams.n_class).to("cuda")
            conf_vals = conf_mat(class_preds, class_targets)
            fig = sns.heatmap(conf_vals.cpu() , annot=True, cmap="Blues", fmt="d")
            wandb.log({f"conf_mat{i}" : wandb.Image(fig)})

            
        return super().test_epoch_end(outputs)

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
    SAVE_SIR = DATASET_DIR+"/Trainings/"+args.save_dir
    os.makedirs(SAVE_SIR, exist_ok=True)
    dataset = dt.GCENgraph(DATASET_DIR)
  
    print(len(dataset))
    train_size = int(0.8 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - (train_size+val_size)
    dataset_train,dataset_valid,dataset_test= torch.utils.data.random_split(dataset, [train_size, val_size,test_size])
    #dataset_test = dt_test.GCENgraph(DATASET_DIR)
    #dataset_valid = MicrographDataValid(DATASET_DIR)
    #dataset_test = MicrographDataValid(DATASET_DIR) # using validation data for testing here
    train_loader = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_dataloader_workers,persistent_workers=True)
    #train_loader = NeighborLoader(data=dataset_train[0], batch_size=BATCH_SIZE, shuffle=True, num_neighbors=30)
    print(train_size)
    
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_dataloader_workers,persistent_workers=True)
    test_loader = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_dataloader_workers,persistent_workers=True)
    #torch.save(test_loader,DATASET_DIR+'/test.pt')
    model = GRNFormerLinkPred(learning_rate=1e-6,encoder_layers=args.encoder_layers)
    
    trainer = pl.Trainer(deterministic=True).from_argparse_args(args)
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss', save_top_k=10, dirpath=args.save_dir, filename='GRNFormer_beeline_{epoch:02d}_{valid_loss:6f}')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopping_callback = EarlyStopping(monitor='valid_loss', mode='min', min_delta=0.0, patience=20)
    trainer.callbacks = [checkpoint_callback, lr_monitor, early_stopping_callback]
    logger = WandbLogger(project=args.project_name, entity=args.entity_name,name=args.save_dir, offline=False, save_dir=SAVE_SIR)
    trainer.logger = logger
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(dataloaders=test_loader, ckpt_path='best')
   



if __name__ == "__main__":
    train_GRNFormerLinkPred()
