import os
import gc
from typing import Any, Callable, Iterable, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC,BinaryAveragePrecision,BinaryConfusionMatrix,BinaryF1Score,BinaryAccuracy,BinaryJaccardIndex,BinaryPrecision,BinaryRecall

from src.models.grnformer.network import *

NODES_DIM = 64
EXP_CHANNEL=759
BERT_CHANNEL=759
EDGE_DIM = 1
NUM_HEADS = 4
ENCODE_LAYERS = 3
OUT_CH=16
DATASET_DIR = os.path.abspath("./")

EPS = 1e-15

OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], Union[torch.optim.lr_scheduler._LRScheduler, ReduceLROnPlateau]]

class GRNFormerLitModule(LightningModule):
    def __init__(
        self,
        # net: torch.nn.Module,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ReduceLROnPlateau,
        node_dim=NODES_DIM,out_chan=OUT_CH,num_heads=NUM_HEADS,edge_dim=EDGE_DIM,encoder_layers=ENCODE_LAYERS
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False) #, ignore=["model", "embed", "edgepred", "reconstruct"])
        
        self.optimizer = optimizer
        self.scheduler = scheduler

        # self.net = net

        # loss function
        self.embed = TransformerAutoencoder(embed_dim=NODES_DIM,nhead=4,num_layers=1)
        self.model = VGAE(encoder=EdgeTransformerEncoder_tcn(node_dim, out_chan, num_head=num_heads,edge_dim=edge_dim,num_layers=encoder_layers),decoder=TransformerDecoder_tcn(latent_dim=out_chan,out_channels=out_chan,num_head=num_heads,num_layers=encoder_layers,edge_dim=edge_dim))
        self.edgepred = EdgePredictor(latent_dim=out_chan)
        self.reconstruct = Reconstruct()
        self.criterian = nn.BCELoss()
        self.loss_fn = self.recon_loss
        self.kl =self.model.kl_loss
        self.metrics = MetricCollection([BinaryAUROC(),BinaryAveragePrecision(),BinaryF1Score(),BinaryAccuracy(),BinaryJaccardIndex(),BinaryPrecision(),BinaryRecall()])
        #self.metrics = self.test
        self.train_metrics_1 = self.metrics.clone(prefix="train_")
        self.train_metrics = self.test
        self.valid_metrics_1 = self.metrics.clone(prefix="valid_")
        self.valid_metrics = self.test
        self.test_metrics_1 = self.metrics.clone(prefix="test_")
        self.test_metrics = self.test
    
    def forward(self, exp_data,bert_data,edge_sampled,edge_attr):

        z = self.embed(exp_data)
        #print(z.shape)
        x = z.squeeze()
        grnnodes_lat, grnedges_lat,mu,logstd= self.model.encode(x,edge_sampled,edge_attr)
        grnnodes_lat,grn_edges,grn_edge_att = self.model.decode(grnnodes_lat,grnedges_lat[0],grnedges_lat[1])
        #print(z.shape,grn_edges.shape)
        edge_prob =self.edgepred(grnnodes_lat,grn_edges,grn_edge_att)
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
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
            #neg_edge_index = sample_embedding_based_negatives(emb,pos_edge_index,pos_edge_index.shape[1]*2,threshold=0.5)
        #print(pos_edge_index,neg_edge_index)
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)
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
        
        #pos_loss = -torch.log(
        #    self.model.decoder(z, pos_edge_index) + EPS).mean()

        
        #neg_loss = -torch.log(1 -
        ##                      self.model.decoder(z, neg_edge_index) +
        #                      EPS).mean()

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
        #from sklearn.metrics import average_precision_score, roc_auc_score
        #if neg_edge_index is None:
        #    neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.reconstruct(z, pos_edge_index,sigmoid=True)
        neg_pred = self.reconstruct(z, neg_edge_index,sigmoid=True)
        preds = torch.cat([pos_pred, neg_pred], dim=0)
        
        #y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        #preds = z.view(-1)
        #num_nodes = z.size(0)
        #y = to_dense_adj(pos_edge_index, max_num_nodes=num_nodes).squeeze(0)
        # Flatten the adjacency matrix and predicted edge probabilities
        y_flat = y
        if prefix=="train":
            metrics = self.train_metrics_1(preds,y_flat.int())
        elif prefix=="valid":
            metrics = self.valid_metrics_1(preds,y_flat.int())
        elif prefix=="test":
            metrics = self.test_metrics_1(preds,y_flat.int())
        else:
            metrics = self.metrics(preds,y_flat.int())
        return metrics
    
    def training_step(self,batch,batch_idx):
        batch_exp_data = batch[0][0].unsqueeze(0).float().cuda()
        
        batch_bert_data = batch[0][0].cuda()
        #print(batch_bert_data.shape)
        batch_edges = batch[0][1].squeeze(0).cuda()
        
        batch_edge_attr =batch[0][2].float().cuda()


        grn_pred_edge,z,mu,logstd = self.forward(batch_exp_data,batch_bert_data,batch_edges,batch_edge_attr)

        batch_targ_pos = batch[0][3].cuda()   
          
        #batch_targ_neg = batch[3].squeeze(0).cuda()
        #batch_tar_mat = batch[4].float().cuda()
        #print(mu,logstd)
        #loss = (self.loss_fn(grn_pred_edge.float(),batch_tar_mat.view(-1))+EPS).mean()
        loss,neg_edge = self.loss_fn(grn_pred_edge,z,batch_targ_pos)
        #loss = loss_pos.mean()+loss_neg.mean()
        klloss = self.kl(mu,logstd)*1/len(z)
        loss=loss+klloss
        #loss = loss + (1 / len(batch_exp_data)) * self.kl()
        
        metrics = self.train_metrics(grn_pred_edge,batch_targ_pos,neg_edge,prefix="train")
        #self.log_dict({"train_auroc":metric_auroc,"train_ap":metric_ap},sync_dist=True)
        self.log_dict(metrics,on_epoch=True,sync_dist=True)
        
        self.log('train_loss',loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    def training_epoch_end(self, outputs):
        # Clear memory or temporary files
        torch.cuda.empty_cache()
        gc.collect()

        #temp_data_path =  tempfile.gettempdir()
        #f os.path.exists(temp_data_path):
        #    os.remove(temp_data_path)

        self.log('memory_cleared', True)

    def validation_step(self,batch,batch_idx):
        batch_exp_data = batch[0][0].unsqueeze(0).float().cuda()
        
        batch_bert_data = batch[0][0].cuda()
        #print(batch_bert_data.shape)
        batch_edges = batch[0][1].squeeze(0).cuda()
        #batch_edge_attr =torch.transpose(batch[0][2],0,1).float().cuda()
        batch_edge_attr =batch[0][2].float().cuda()
        #print(batch_edge_attr)
        #print(batch_edges.shape,batch_edge_attr.shape)
        batch_targ_pos = batch[0][3].cuda()  
        
        grn_pred_edge,z,mu,logstd = self.forward(batch_exp_data,batch_bert_data,batch_edges,batch_edge_attr)
        #print(grn_pred_edge.shape)
        #print(batch_targ_pos.shape)
        #loss = (self.loss_fn(grn_pred_edge.float(),batch_tar_mat.view(-1))+EPS).mean()
        loss,neg_edge = self.loss_fn(grn_pred_edge,z,batch_targ_pos)
        klloss = self.kl(mu,logstd)*1/len(z)
        loss=loss+klloss
        #loss = loss_pos.mean()+loss_neg.mean()
        #loss = loss + (1 / len(batch_exp_data)) * self.kl()
        #loss = loss + (1 / len(batch_data)) * self.kl()
        metrics = self.valid_metrics(grn_pred_edge,batch_targ_pos,neg_edge,prefix="valid")
        self.log_dict(metrics,sync_dist=True,batch_size=1,on_epoch=True)
        #self.log_dict({"valid_auroc":metric_auroc,"valid_ap":metric_ap},sync_dist=True)
        self.log('valid_loss',loss, on_step=True, on_epoch=True, sync_dist=True)
       
    
    def test_step(self,batch, batch_idx):
        batch_exp_data = batch[0][0].unsqueeze(0).float().cuda()
        
        batch_bert_data = batch[0][0].cuda()
        #print(batch_bert_data.shape)
        batch_edges = batch[0][1].squeeze(0).cuda()
        batch_edge_attr =batch[0][2].float().cuda()


        grn_pred_edge,z,mu,logstd  = self.forward(batch_exp_data,batch_bert_data,batch_edges,batch_edge_attr)
        batch_targ_pos = batch[0][3].cuda()  
        
        #loss = (self.loss_fn(grn_pred_edge.float(),batch_tar_mat.view(-1))+EPS).mean()
        loss,neg_edge = self.loss_fn(grn_pred_edge,z,batch_targ_pos)
        
        klloss = self.kl(mu,logstd)*1/len(z)
        loss=loss+klloss
        #loss = loss_pos.mean()+loss_neg.mean()
        #loss = loss + (1 / len(batch_exp_data)) * self.kl()
        #loss = loss + (1 / len(batch_data)) * self.kl()
        metrics = self.test_metrics(grn_pred_edge,batch_targ_pos,neg_edge_index=neg_edge,prefix="test")
        #self.log_dict({"test_auroc":metric_auroc,"test_ap":metric_ap},sync_dist=True)
        self.log_dict(metrics,sync_dist=True,on_epoch=True)
       
        #conf_mat = BinaryConfusionMatrix().to("cuda")
        #conf_vals = conf_mat(grn_pred_edge,batch_tar_mat.view(-1).int())
        #print("Test Data Confusion Matrix: \n")
        #print(conf_vals)
        
        self.log('test_loss',loss, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss",
                "interval": "epoch",
                "strict": True,
                "frequency": 1,
            },
        }
