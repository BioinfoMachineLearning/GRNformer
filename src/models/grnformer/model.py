import os
import gc
from typing import Any, Callable, Iterable, Union

import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC,BinaryAveragePrecision,BinaryConfusionMatrix,BinaryF1Score,BinaryAccuracy,BinaryJaccardIndex,BinaryPrecision,BinaryRecall
from torch_geometric.utils import negative_sampling
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
        totalnodes = 0, tf_file="", exp_file="", net_file="", output_file="", node_dim=NODES_DIM,out_chan=OUT_CH,num_heads=NUM_HEADS,edge_dim=EDGE_DIM,encoder_layers=ENCODE_LAYERS
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False) #, ignore=["model", "embed", "edgepred", "reconstruct"])
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.totalnodes = totalnodes
        self.tf_file = tf_file
        self.exp_file = exp_file
        self.net_file = net_file
        self.output_file = output_file

        # self.net = net

        # loss function
        self.embed = TransformerAutoencoder(embed_dim=NODES_DIM,nhead=4,num_layers=1)
        self.model = VGAE(encoder=EdgeTransformerEncoder_tcn(node_dim, out_chan, num_head=num_heads,edge_dim=edge_dim,num_layers=encoder_layers),decoder=TransformerDecoder_tcn(latent_dim=out_chan,out_channels=out_chan,num_head=num_heads,num_layers=encoder_layers,edge_dim=edge_dim))
        self.edgepred = EdgePredictor(latent_dim=out_chan)
        self.reconstruct = Reconstruct()
        self.criterian = nn.BCELoss()
        self.loss_fn = self.recon_loss
        self.kl =self.model.kl_loss
        self.metrics = MetricCollection([BinaryAUROC(),BinaryAveragePrecision(),BinaryF1Score(threshold=0.1),BinaryAccuracy(threshold=0.1),BinaryJaccardIndex(threshold=0.1),BinaryPrecision(threshold=0.1),BinaryRecall(threshold=0.1)])
        #self.metrics = self.test
        self.train_metrics_1 = self.metrics.clone(prefix="train_")
        self.train_metrics = self.test
        self.valid_metrics_1 = self.metrics.clone(prefix="valid_")
        self.valid_metrics = self.test
        self.test_metrics_1 = self.metrics.clone(prefix="test_")
        self.test_metrics = self.test
        self.outputs = []
    
    def forward(self, exp_data,edge_sampled,edge_attr):

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
        if pos_edge_index.numel() > 0:
            
            if neg_edge_index is None:
                neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
                #neg_edge_index = sample_embedding_based_negatives(emb,pos_edge_index,pos_edge_index.shape[1]*2,threshold=0.5)
            #print(pos_edge_index,neg_edge_index)
            pos_y = z.new_ones(pos_edge_index.size(1))

            neg_y = z.new_zeros(neg_edge_index.size(1))
            y = torch.cat([pos_y, neg_y], dim=0)
            pos_pred = self.reconstruct(z, pos_edge_index,sigmoid=True)
            print(pos_pred)
            neg_pred = self.reconstruct(z, neg_edge_index,sigmoid=True)
            pred = torch.cat([pos_pred, neg_pred], dim=0)

            loss = self.criterian(pred,y)

            return loss,neg_edge_index
        else:
            return torch.tensor(0.0).to(z.device),neg_edge_index
    
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
        from sklearn.metrics import average_precision_score, roc_auc_score,precision_recall_curve

        #print(preds.shape,y.shape,y_flat.shape)
        if prefix=="train":
            metrics = self.train_metrics_1(preds,y_flat.int())
            return metrics
        elif prefix=="valid":
            metrics = self.valid_metrics_1(preds,y_flat.int())
            return metrics
        elif prefix=="test":
            total_metrics = {}
           
            for i in range(0,100):
                #if neg_edge_index is None:
                neg_edge_index = negative_sampling(pos_edge_index, z.size(0))#self.sample_negative_edges(z.size(0), pos_edge_index, pos_edge_index.size(1))
                pos_y = z.new_ones(pos_edge_index.size(1))
                neg_y = z.new_zeros(neg_edge_index.size(1))
                y = torch.cat([pos_y, neg_y], dim=0)
                #print(neg_edge_index)
                pos_pred = self.reconstruct(z, pos_edge_index,sigmoid=True)
                neg_pred = self.reconstruct(z, neg_edge_index,sigmoid=True)
                preds = torch.cat([pos_pred, neg_pred], dim=0)
                #preds =(z).view(-1).cuda()
                # #y, preds = y.detach().cpu(), preds.detach().cpu()
        
                #num_nodes = z.size(0)
                #y = to_dense_adj(pos_edge_index, max_num_nodes=num_nodes).squeeze(0)
                y_flat = y.cuda()
                metrics = self.test_metrics_1(preds,y_flat.int())
                #precision, recall, thresholds = precision_recall_curve(y_true=y_flat.detach().cpu(), y_score=preds.detach().cpu())
                conf_mat = BinaryConfusionMatrix(threshold=0.5)
                conf_vals = conf_mat(preds.detach().cpu(),y_flat.detach().cpu())
                #use all the metrics to calculate the average metrics
                #print(metrics)
                # Automatically accumulate all metrics
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value

            # Calculate average metrics
            average_metrics = {key: (value / 100) for key, value in total_metrics.items()}
            return average_metrics,conf_vals,pos_pred#,precision,recall,thresholds
        else:
            metrics = self.metrics(preds,y_flat.int())
            return metrics
    
    def training_step(self,batch,batch_idx):
        batch_exp_data = batch[0][0].unsqueeze(0).float().cuda()
        
        
        #print(batch_bert_data.shape)
        batch_edges = batch[0][1].squeeze(0).cuda()
        
        batch_edge_attr =batch[0][2].float().cuda()


        grn_pred_edge,z,mu,logstd = self.forward(batch_exp_data,batch_edges,batch_edge_attr)

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
        
        
        #print(batch_bert_data.shape)
        batch_edges = batch[0][1].squeeze(0).cuda()
        #batch_edge_attr =torch.transpose(batch[0][2],0,1).float().cuda()
        batch_edge_attr =batch[0][2].float().cuda()
        #print(batch_edge_attr)
        #print(batch_edges.shape,batch_edge_attr.shape)
        batch_targ_pos = batch[0][3].cuda()  
        
        grn_pred_edge,z,mu,logstd = self.forward(batch_exp_data,batch_edges,batch_edge_attr)
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
        
        
        batch_edges = batch[0][1].squeeze(0).cuda()
        batch_edge_attr = torch.transpose(batch[0][2].unsqueeze(0),0,1).float().cuda()


        grn_pred_edge,z,mu,logstd  = self.forward(batch_exp_data,batch_edges,batch_edge_attr)
        batch_targ_pos = batch[0][3].cuda()  
        
        loss,neg_edge = self.loss_fn(grn_pred_edge,z,batch_targ_pos)
        
        klloss = self.kl(mu,logstd)*1/len(z)
        loss=loss+klloss

        self.outputs.append({f'preds_grn' : grn_pred_edge,f'target_grnindex':batch_targ_pos,f'nodemap':batch[0][4]})
    
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

    def on_test_epoch_end(self):
        # Log individual results for each dataset
        outputs = self.outputs
        print(len(outputs))
        num_nodes = self.totalnodes
        link_prediction_matrix = torch.zeros((num_nodes, num_nodes)).cuda()
        counts_matrix = torch.zeros((num_nodes, num_nodes)).cuda()
        all_edges=[]
        #if args.TFspecies=="human":
        tf_genes  = pd.read_csv(os.path.abspath(self.tf_file),header=None)
        tf_genes.columns = ['TFs']
        #elif args.TFspecies=="mouse":
        #    tf_genes = pd.read_csv(os.path.dirname(os.path.abspath(args.exp_file))+"/TFs.csv")['TF'].to_list()
        tf_weight = 1  # Weight for TF genes
        # Assuming `outputs` is a list of dictionaries, and link_prediction_matrix and counts_matrix are predefined tensors
        expression_data = pd.read_csv(self.exp_file,index_col=0)
        
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
            # #Create a mask for pro
            # babilities greater than 0.5
            mask = batch_probs > 0.3

            tf_indices = [index for index, gene in enumerate(gene_names) if gene in tf_genes['TFs'].to_list()]  # Assuming `tf_genes` is a list of TF gene names
            non_tf_indices = [index for index, gene in enumerate(gene_names) if gene not in tf_genes]
            # Convert tf_indices and non_tf_indices to tensors for compatibility with indexing
            tf_indices = torch.tensor(tf_indices, dtype=torch.long).cuda()

            
            # Separate masked batch indices and probabilities
            masked_batch_indices_0 = batch_indices[0][mask]
            masked_batch_indices_1 = batch_indices[1][mask]
            masked_batch_probs = batch_probs[mask]

            # Determine if masked_batch_indices contain TF genes
            is_source_tf = torch.isin(masked_batch_indices_0, tf_indices)
            is_target_tf = torch.isin(masked_batch_indices_1, tf_indices)
            is_tf_edge = is_source_tf  # True if source is TF gene

            # Create weights based on TF presence
            tf_weights = torch.where(is_tf_edge, tf_weight, 0.0)  # Apply tf_weight where either node is a TF

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
        truedges = pd.read_csv(self.net_file)
        truedges = truedges[truedges.Gene1.isin(tf_genes['TFs'])]
        #get indices of true edges based on gene names
        true_edges = [(gene_names.index(edge[0]), gene_names.index(edge[1])) for edge in truedges.values]
        #convert true edges to the similar format as unique edges
        true_edges_list = [list(edge) for edge in true_edges]
        #convert trueedges to tensor to match the format of unique edges
        true_edges_tensor = torch.tensor(true_edges_list).t()
        # Convert to a tensor and transpose to match edge_index format
        edge_index_unique = torch.tensor(unique_edges_list).t().cuda()
        #edge_index present where source is  tf indices
        
        edge_index_unique = edge_index_unique[:, torch.isin(edge_index_unique[0], tf_indices)]

        print(true_edges_tensor.shape,edge_index_unique.shape)
        #print(edge_index_unique.shape)
        # Avoid division by zero by adding a small epsilon
        counts_matrix[counts_matrix == 0] = 1e-8
        #print(link_prediction_matrix,counts_matrix)
        # Compute final predictions using weighted average
        link_prediction_matrix=link_prediction_matrix.fill_diagonal_(0)
        final_predictions =(link_prediction_matrix / counts_matrix)
        #min_max normalization
        final_predictions = (final_predictions - final_predictions.min()) / (final_predictions.max() - final_predictions.min())

        df = pd.DataFrame(final_predictions.cpu().numpy())

        metrics,conf_val,preds = self.test_metrics(final_predictions,edge_index_unique,prefix="test")
        #print(thresholds)
        print(metrics)
        #metrics, conf_val, preds = self.test_metrics(final_predictions, edge_index_unique, prefix="test")
        metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
        #remove the prefix from the metric names
        metrics_df['Metric'] = metrics_df['Metric'].apply(lambda x: x.replace("test_",""))
        #rename Average Precision to AUPRC
        metrics_df['Metric'] = metrics_df['Metric'].apply(lambda x: "AUPRC" if x=="Average Precision" else x)
        #convert metric values to float, from tensor with only 3 decimal points
        metrics_df['Value'] = metrics_df['Value'].apply(lambda x: x.item())

        metrics_df.to_csv(os.path.abspath((self.output_file)).split('.')[0] + "-metrics.csv", index=False,float_format='%.3f')
        # Log final predictions (or return if needed)
        final_predictions1,gene_names = self.add_index_and_column(self.exp_file, df)
        #print(final_predictions1)
        
        # Filter predictions with probability greater than 0.5
        filtered_predictions = final_predictions1[final_predictions1>-0.01].stack().reset_index()
        filtered_predictions.columns = ['gene1', 'gene2', 'probability']
        # select only rows with gene1 as TF genes
        
        sorted_df = filtered_predictions.sort_values(by='probability', ascending=False)

        top_k_rows = sorted_df
        #round the probability values to 2 decimal precision
        top_k_rows['probability'] = top_k_rows['probability'].round(3)
        #assign new column name of rank to the dataframe based on probability values of 2 decimal precision with same proabibilty having same rank
        top_k_rows['rank'] = top_k_rows['probability'].rank(method='dense',ascending=False).astype(int)
        #select  ranks of length unique_edges
        #top_k_rows = top_k_rows[top_k_rows['rank'] <= len(unique_edges)]
        print(top_k_rows)
        # Save filtered predictions to CSV file
        top_k_rows.to_csv(os.path.abspath(self.output_file), index=False)

        
        return final_predictions

    def predict_step(self, batch, batch_idx):
        batch_exp_data = batch[0][0].unsqueeze(0).float().cuda()
        
        
        batch_edges = batch[0][1].squeeze(0).cuda()
        batch_edge_attr = torch.transpose(batch[0][2].unsqueeze(0),0,1).float().cuda()
        grn_pred_edge,z,mu,logstd  = self(batch_exp_data,batch_edges,batch_edge_attr)
        self.outputs.append({f'preds_grn' : grn_pred_edge,f'nodemap':batch[0][4]})
        return grn_pred_edge
    def on_predict_epoch_end(self):
        # Log individual results for each dataset
        outputs = self.outputs
        num_nodes = self.totalnodes
        link_prediction_matrix = torch.zeros((num_nodes, num_nodes)).cuda()
        counts_matrix = torch.zeros((num_nodes, num_nodes)).cuda()
        all_edges=[]
        #if args.TFspecies=="human":
        tf_genes  = pd.read_csv(os.path.abspath(self.tf_file),header=None)
        tf_genes.columns = ['TFs']
        #elif args.TFspecies=="mouse":
        #    tf_genes = pd.read_csv(os.path.dirname(os.path.abspath(args.exp_file))+"/TFs.csv")['TF'].to_list()
        tf_weight = 1
        # Assuming `outputs` is a list of dictionaries, and link_prediction_matrix and counts_matrix are predefined tensors        
        expression_data = pd.read_csv(self.exp_file,index_col=0)
        # Get the gene names from the index of the expression file
        gene_names = expression_data.index.tolist()
        for sam in range(len(outputs)):
            # Extract current sample data
            dataset_outputs = outputs[sam]['preds_grn']
            #target_grn = outputs[sam]['target_grnindex']
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
            # #Create a mask for pro
            # babilities greater than 0.5
            mask = batch_probs > 0.3

            tf_indices = [index for index, gene in enumerate(gene_names) if gene in tf_genes['TFs'].to_list()]  # Assuming `tf_genes` is a list of TF gene names
            non_tf_indices = [index for index, gene in enumerate(gene_names) if gene not in tf_genes]
            # Convert tf_indices and non_tf_indices to tensors for compatibility with indexing
            tf_indices = torch.tensor(tf_indices, dtype=torch.long).cuda()

            
            # Separate masked batch indices and probabilities
            masked_batch_indices_0 = batch_indices[0][mask]
            masked_batch_indices_1 = batch_indices[1][mask]
            masked_batch_probs = batch_probs[mask]

            # Determine if masked_batch_indices contain TF genes
            is_source_tf = torch.isin(masked_batch_indices_0, tf_indices)
            is_target_tf = torch.isin(masked_batch_indices_1, tf_indices)
            is_tf_edge = is_source_tf  # True if source is TF gene

            # Create weights based on TF presence
            tf_weights = torch.where(is_tf_edge, tf_weight, 0.0)  # Apply tf_weight where either node is a TF

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
        counts_matrix[counts_matrix == 0] = 1e-8
        #print(link_prediction_matrix,counts_matrix)
        # Compute final predictions using weighted average
        link_prediction_matrix=link_prediction_matrix.fill_diagonal_(0)
        final_predictions =(link_prediction_matrix / counts_matrix)
        #min_max normalization
        final_predictions = (final_predictions - final_predictions.min()) / (final_predictions.max() - final_predictions.min())

        df = pd.DataFrame(final_predictions.cpu().numpy())
        final_predictions1,gene_names = self.add_index_and_column(self.exp_file, df)
        #print(final_predictions1)
        
        # Filter predictions with probability greater than 0.5
        filtered_predictions = final_predictions1[final_predictions1>-0.01].stack().reset_index()
        filtered_predictions.columns = ['gene1', 'gene2', 'probability']
        # select only rows with gene1 as TF genes
        
        sorted_df = filtered_predictions.sort_values(by='probability', ascending=False)

        top_k_rows = sorted_df
        #round the probability values to 2 decimal precision
        top_k_rows['probability'] = top_k_rows['probability'].round(3)
        #assign new column name of rank to the dataframe based on probability values of 2 decimal precision with same proabibilty having same rank
        top_k_rows['rank'] = top_k_rows['probability'].rank(method='dense',ascending=False).astype(int)
        #select  ranks of length unique_edges
        #top_k_rows = top_k_rows[top_k_rows['rank'] <= len(unique_edges)]
        print(top_k_rows)
        # Save filtered predictions to CSV file
        top_k_rows.to_csv(os.path.abspath(self.output_file), index=False)
        return final_predictions
    
    
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
