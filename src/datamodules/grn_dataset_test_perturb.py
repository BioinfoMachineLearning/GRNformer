import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.utils import from_networkx
import networkx as nx

from torch_geometric.loader import NeighborSampler
# from torch.utils.data import DataLoader
import os
from itertools import chain
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.loader import DataListLoader
from torch_geometric.sampler import NumNeighbors
from torch_geometric.utils import dense_to_sparse, k_hop_subgraph,to_dense_adj,degree
import random
class GeneExpressionDataset(InMemoryDataset):
    def __init__(self, root, gene_expression_file, tf_genes, regulation_file, 
             coexpression_threshold=0.5, max_subgraph_size=10, 
             transform=None, pre_transform=None,
             noise_std=0.0, dropout_fraction=0.0, random_seed=None,
             perturb_after_normalization=False, scale_noise=True):
        self.gene_expression_file = gene_expression_file
        self.tf_genes = tf_genes
        self.regulation_file = regulation_file
        self.coexpression_threshold = coexpression_threshold
        self.max_subgraph_size = max_subgraph_size
        self.noise_std = noise_std
        self.dropout_fraction = dropout_fraction
        self.random_seed = random_seed
        self.perturb_after_normalization = perturb_after_normalization
        self.scale_noise = scale_noise
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            random.seed(random_seed)
        super(GeneExpressionDataset, self).__init__(root, transform, pre_transform)
        self.data_list = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        # Patch: look for files directly in root, not in 'raw/'
        return [self.gene_expression_file, self.regulation_file]

    @property
    def raw_paths(self):
        # Patch: return full paths to the raw files in the root directory
        return [os.path.join(self.root, f) for f in self.raw_file_names]

    @property
    def processed_file_names(self):
        # Include perturbation parameters in filename for proper caching
        perturb_suffix = ""
        if self.noise_std > 0.0 or self.dropout_fraction > 0.0:
            # Format floats to avoid issues with decimal points in filenames
            noise_str = f"{self.noise_std:.3f}".replace('.', 'p')
            dropout_str = f"{self.dropout_fraction:.3f}".replace('.', 'p')
            scale_str = "scaled" if self.scale_noise else "abs"
            perturb_suffix = f"_noise{noise_str}_dropout{dropout_str}_{scale_str}"
            if self.perturb_after_normalization:
                perturb_suffix += "_postnorm"
            if self.random_seed is not None:
                perturb_suffix += f"_seed{self.random_seed}"
        return [f"inference_grid_test_max_{self.max_subgraph_size}_{self.coexpression_threshold}{perturb_suffix}.pt"]

    def download(self):
        pass
    def istf(self,genes,TFs):
        #Read human TFs list,and add a bool feature if a gene is a TF or not
        

        is_tf = genes.str.upper().isin(TFs)  
        #print(is_tf)
        return is_tf 
    def construct_networkx(self, edge_weight_matrix, gene_exp_matr):
        """
        Constructs a network from an edge weight matrix.

        Args:
            edge_weight_matrix: A square matrix of edge weights.

        Returns:
            A `torch_geometric.data.Data` object representing the network.
        """
        # Use configurable coexpression threshold instead of hardcoded 0.5
        edge_index = (abs(edge_weight_matrix) > self.coexpression_threshold).nonzero().t()
        row, col = edge_index
        edge_weight = edge_weight_matrix[row, col]
        
        # Create a Data object to represent the graph.
        data = Data(x=torch.tensor(gene_exp_matr), edge_index=edge_index, edge_weight=edge_weight)
        
        return data




    def activate_grn(self,exp_mat,exp_mean, target_adj,is_tf):

       for i in range(len(target_adj)):

              if ((exp_mat[i] < exp_mean[i]) and (is_tf[i]==1)):

                     target_adj[i] = 0

       return target_adj
    def min_max(self,tensor):
        # Compute the min and max values for each row
        row_min, _ = torch.min(tensor, dim=1, keepdim=True)
        row_max, _ = torch.max(tensor, dim=1, keepdim=True)
        print(row_max.shape)
        # Compute the range, avoid division by zero by replacing 0 with 1 in the range
        row_range = torch.where(row_max != row_min, row_max - row_min, torch.ones_like(row_max))

        # Apply Min-Max normalization
        normalized_tensor = (tensor - row_min) / row_range

        return normalized_tensor
    def min_max_dim0(self,exp_mat):
        max_n,_ = torch.max(exp_mat,dim=0)

        min_n,_ = torch.min(exp_mat,dim=0)
        exp_mat = (exp_mat - min_n ) / (max_n - min_n)
        return exp_mat.T
    def z_score(self,exp_mat):
            #exp_mat = torch.log(expression_mat)
            mean_sam = torch.mean(exp_mat, axis = 1,keepdim=False)
            std_sam = torch.mean(exp_mat, axis = 1,keepdim=False)

            z_score = (exp_mat - mean_sam)/std_sam

            return z_score
    def z_score_per_cell(self, exp_mat):
        # exp_mat: n x m matrix (n = number of genes, m = number of cells)

        # Calculate the mean of each column (cell)
        mean_per_cell = torch.mean(exp_mat, axis=0, keepdim=False)  # shape: (m,)

        # Calculate the standard deviation of each column (cell)
        std_per_cell = torch.std(exp_mat, axis=0, keepdim=False)    # shape: (m,)

        # Avoid division by zero by adding a small constant to std
        epsilon = 1e-8
        z_score = (exp_mat - mean_per_cell) / (std_per_cell + epsilon)

        return z_score
    
    def add_gaussian_noise(self, exp_mat, noise_std):
        """
        Add Gaussian noise to gene expression matrix.
        
        Args:
            exp_mat: numpy array or torch tensor of gene expression values
            noise_std: standard deviation of Gaussian noise to add
            
        Returns:
            Perturbed expression matrix
        """
        if noise_std <= 0.0:
            return exp_mat
        
        if isinstance(exp_mat, np.ndarray):
            noise = np.random.normal(0, noise_std, exp_mat.shape)
            return exp_mat + noise
        else:
            noise = torch.randn_like(exp_mat) * noise_std
            return exp_mat + noise
    
    def apply_dropout(self, exp_mat, dropout_fraction):
        """
        Randomly drop (set to zero) a fraction of gene expression values.
        
        Args:
            exp_mat: numpy array or torch tensor of gene expression values
            dropout_fraction: fraction of values to drop (0.0 to 1.0)
            
        Returns:
            Expression matrix with dropped values set to zero
        """
        if dropout_fraction <= 0.0:
            return exp_mat
        
        if isinstance(exp_mat, np.ndarray):
            mask = np.random.random(exp_mat.shape) > dropout_fraction
            return exp_mat * mask
        else:
            mask = (torch.rand_like(exp_mat) > dropout_fraction).float()
            return exp_mat * mask

    
    def process(self):
        
        self.data_list = []
        self.all_sampled_indices=[]
        self.all_edges=[]
        self.node_idx_maps=[]
        # Step 1: Create a weighted gene co-expression network
        gene_expression_data = pd.read_csv(self.raw_paths[0], index_col=0)
        gene_expression_data = gene_expression_data.apply(np.arcsinh)
        
        # Store original data statistics for scaling perturbations
        exp_values_original = gene_expression_data.values.copy()
        exp_mean = np.mean(exp_values_original)
        exp_std = np.std(exp_values_original)
        exp_abs_max = np.abs(exp_values_original).max()
        
        # Apply perturbations if specified
        if self.noise_std > 0.0 or self.dropout_fraction > 0.0:
            print(f"Applying perturbations: noise_std={self.noise_std}, dropout_fraction={self.dropout_fraction}")
            print(f"Data statistics: mean={exp_mean:.4f}, std={exp_std:.4f}, abs_max={exp_abs_max:.4f}")
            exp_values = exp_values_original.copy()
            
            # Apply dropout first (before noise, as dropout sets values to zero)
            if self.dropout_fraction > 0.0:
                exp_values = self.apply_dropout(exp_values, self.dropout_fraction)
                num_dropped = np.sum(exp_values == 0)
                total_values = exp_values.size
                print(f"Dropped {self.dropout_fraction*100:.1f}% of expression values ({num_dropped}/{total_values} values)")
            
            # Apply Gaussian noise
            if self.noise_std > 0.0:
                if self.scale_noise:
                    # Scale noise by data standard deviation to make it meaningful
                    # noise_std is interpreted as a fraction of data std (e.g., 0.1 = 10% of data std)
                    actual_noise_std = self.noise_std * exp_std
                    exp_values = self.add_gaussian_noise(exp_values, actual_noise_std)
                    print(f"Added Gaussian noise with std={actual_noise_std:.4f} ({self.noise_std*100:.1f}% of data std={exp_std:.4f}) [SCALED]")
                else:
                    # Use noise_std as absolute value
                    exp_values = self.add_gaussian_noise(exp_values, self.noise_std)
                    print(f"Added Gaussian noise with std={self.noise_std:.4f} (absolute value)")
                    print(f"  Data std={exp_std:.4f}, Noise as % of data std={100*self.noise_std/exp_std:.2f}% [ABSOLUTE]")
            
            # Update the dataframe with perturbed values
            gene_expression_data = pd.DataFrame(exp_values, 
                                               index=gene_expression_data.index,
                                               columns=gene_expression_data.columns)
            
            # Verify perturbations were applied
            perturbed_mean = np.mean(exp_values)
            perturbed_std = np.std(exp_values)
            print(f"After perturbation: mean={perturbed_mean:.4f}, std={perturbed_std:.4f}")
            print(f"Change in mean: {perturbed_mean - exp_mean:.4f}, Change in std: {perturbed_std - exp_std:.4f}")
        
        correlation_matrix = torch.as_tensor(np.array(np.corrcoef(gene_expression_data.values)))
        num_genes = correlation_matrix.shape[0]
        print(correlation_matrix.shape)
        # Create edge index and edge weight tensors
        edge_index, edge_weight = dense_to_sparse(correlation_matrix)
        print(edge_index.shape,edge_weight.shape)
        # Create the PyTorch Geometric Data object
        full_graph = self.construct_networkx(correlation_matrix,gene_expression_data.values)
        #full_graph = Data(x=torch.tensor(gene_expression_data.values, dtype=torch.float),
        #                  edge_index=edge_index,
        #                  edge_weight=edge_weight)
        print(full_graph)
        gene_indices = {gene: idx for idx, gene in enumerate(gene_expression_data.index)}
        regulation_data = pd.read_csv(self.raw_paths[1])
        # Check if the regulation_data DataFrame has a 'Type' column (case-insensitive) and filter for positive (Type == 1) only
        # Process the regulation_data to ensure -1 and 1 become label 1, and others become 0
        # for both 'Type' or 'Label' columns if present

        # Identify the label column: 'Type' or 'Label'
        label_col = None
        for col in regulation_data.columns:
            if col.lower() == 'type':
                label_col = col
                break
        if label_col is None:
            for col in regulation_data.columns:
                if col.lower() == 'label':
                    label_col = col
                    break

        if label_col is not None:
            # Convert values: if value is 1 or -1 assign 1, else 0
            regulation_data['__custom_bin_label__'] = regulation_data[label_col].apply(lambda x: 1 if x in [1, -1] else 0)
        else:
            # If neither present, we can't binarize, just skip binarization (or could raise error)
            regulation_data['__custom_bin_label__'] = 0

        regulation_matrix = np.zeros((num_genes, num_genes))
        genes = gene_expression_data.index
        Exp_tar_adj = pd.DataFrame(index=gene_indices.keys(), columns=gene_indices.keys())
        # Fill in the values based on regulation_data
        for index, row in regulation_data.iterrows():
            if (row['Gene1'] in genes) and (row['Gene2'] in genes):
                Exp_tar_adj.at[row['Gene1'], row['Gene2']] = row['__custom_bin_label__']
        regulation_matrix = torch.as_tensor(Exp_tar_adj.fillna(0).to_numpy())
        Is_tf = torch.as_tensor(self.istf(genes,self.tf_genes))
        print(Is_tf.sum())
        #full_graph.x = self.z_score_per_cell(full_graph.x)
        
        #print(regulation_matrix.loc["CEBPB",tf_gene],gene_indices["CEBPB"])
        print(regulation_matrix.shape)
        # Step 2: Sample subgraphs
        for tf_gene in self.tf_genes:
            if tf_gene in gene_indices:
                tf_idx = gene_indices[tf_gene]
                print(tf_idx)
                #print(correlation_matrix[tf_idx])
                
                # Get 1-hop and 2-hop neighbors
                num_hops = 1
                while num_hops<10:
                    subgraph_node_idx1, subgraph_edge_index1, _, _ = k_hop_subgraph(tf_idx, num_hops=num_hops, edge_index=full_graph.edge_index, num_nodes=num_genes)
                    if len(subgraph_node_idx1) >= self.max_subgraph_size:
                        break
                    #print(num_hops,len(subgraph_node_idx))
                    num_hops += 1
                
                #if len(subgraph_node_idx) < 500:
                #    continue
                #subgraph_node_idx, subgraph_edge_index, _, _ = k_hop_subgraph(tf_idx, num_hops=2, edge_index=full_graph.edge_index, num_nodes=num_genes)
                #print(subgraph_edge_index)
                first_hop_neighbors = full_graph.edge_index[1][full_graph.edge_index[0] == tf_idx]
                first_hop_neighbors = first_hop_neighbors.unique()
                #print(len(first_hop_neighbors))
                first_hop_neighbors = first_hop_neighbors[first_hop_neighbors!= tf_idx]
                first_hop_neighbors = first_hop_neighbors[torch.randperm(len(first_hop_neighbors))]
                #print(len(first_hop_neighbors))
                second_hop_neighbors = subgraph_node_idx1[~torch.isin(subgraph_node_idx1,first_hop_neighbors) & (subgraph_node_idx1 != tf_idx)]                
                max_nodes = self.max_subgraph_size
                num_first_hop = len(first_hop_neighbors)
                chunk_size = max_nodes - 1  # Reserve 1 slot for `tf_idx`
                chunks = [first_hop_neighbors[i:i + chunk_size] for i in range(0, num_first_hop, chunk_size)]
                print("chunks:",len(chunks))
                for chunk in chunks:
                    if len(chunk) + 1 >= max_nodes:
                        subgraph_node_idx = torch.cat([torch.tensor([tf_idx]), torch.tensor(chunk[:chunk_size])])
                    else:
                        remaining_nodes = max_nodes - (len(chunk) + 1)
                        subgraph_node_idx = torch.cat([torch.tensor([tf_idx]), torch.tensor(chunk), second_hop_neighbors[:remaining_nodes]])
                
                    subgraph_node_set = set(subgraph_node_idx.tolist())

                    #print(torch.isin(tf_idx,subgraph_node_idx))
                    # Remap the subgraph node indices
                    subgraph_node_idx = subgraph_node_idx.unique()
                    #print(subgraph_node_idx)
                    #print(subgraph_edge_index1.shape)
                    
                    #print(len(subgraph_node_idx))
                    node_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph_node_idx.tolist())}
                    reverse_mapping = {v: k for k, v in node_idx_map.items()}
                    # Filter edges to keep only those within the subgraph nodes
                    mask = torch.isin(subgraph_edge_index1[0], subgraph_node_idx) & torch.isin(subgraph_edge_index1[1], subgraph_node_idx)
                    subgraph_edge_index = subgraph_edge_index1[:, mask]
                    subgraph_edge_index = torch.tensor([[node_idx_map[i.item()], node_idx_map[j.item()]] for i, j in zip(subgraph_edge_index[0], subgraph_edge_index[1])], dtype=torch.long).t().contiguous()
                    # Get the corresponding edge weights for the filtered edges
                    subgraph_edge_weight = full_graph.edge_weight[
                        torch.isin(full_graph.edge_index[0], subgraph_node_idx) & 
                        torch.isin(full_graph.edge_index[1], subgraph_node_idx)
                    ]
                    #print(subgraph_edge_index)
                    exp_x = full_graph.x[subgraph_node_idx]
                    #minm_x = self.min_max(exp_x)
                    zscore = self.z_score_per_cell(exp_x)
                    
                    # Apply perturbations after normalization if requested
                    if self.perturb_after_normalization and (self.noise_std > 0.0 or self.dropout_fraction > 0.0):
                        zscore_np = zscore.numpy() if isinstance(zscore, torch.Tensor) else zscore
                        if self.dropout_fraction > 0.0:
                            zscore_np = self.apply_dropout(zscore_np, self.dropout_fraction)
                        if self.noise_std > 0.0:
                            if self.scale_noise:
                                # For normalized data, scale noise relative to normalized data std
                                # Since normalized data has std ~1, this makes noise_std directly interpretable
                                zscore_std = np.std(zscore_np)
                                actual_noise_std = self.noise_std * zscore_std
                                zscore_np = self.add_gaussian_noise(zscore_np, actual_noise_std)
                            else:
                                # Use noise_std as absolute value
                                zscore_np = self.add_gaussian_noise(zscore_np, self.noise_std)
                        zscore = torch.tensor(zscore_np, dtype=torch.float32) if isinstance(zscore, torch.Tensor) else zscore_np
                    
                    #print("zscore shape",zscore.shape)
                    subgraph_x = torch.column_stack((zscore,Is_tf[subgraph_node_idx]))

                    #subgraph_x = full_graph.x[subgraph_node_idx]
                    if len(subgraph_node_idx) < 3:
                        print(f"Skipping degenerate subgraph with {len(subgraph_node_idx)} nodes (TF branch)")
                        continue
                    subgraph_data = Data(x=subgraph_x, edge_index=subgraph_edge_index, edge_weight=subgraph_edge_weight)
                    # #print(subgraph_data.edge_weight.shape,common_neigh[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1).shape)
                    edge_weight1 = torch.cat([subgraph_data.edge_weight.unsqueeze(1)])

                    subgraph_data.edge_weight = edge_weight1
    
                    # Process the regulation matrix for the subgraph
                    subgraph_indices = subgraph_node_idx.tolist()
                    subgraph_regulation_matrix = regulation_matrix[:,subgraph_indices] [subgraph_indices,:]
                    #print(subgraph_regulation_matrix,regulation_matrix.sum())
                    subgraph_regulation_edge_index, subgraph_regulation_edge_weight = dense_to_sparse(torch.tensor(subgraph_regulation_matrix))
                    old_edge_indices = torch.zeros_like(subgraph_regulation_edge_index)#.cuda()
                    for m in range(subgraph_regulation_edge_index.size(1)):  # Iterate over all edges
                         old_edge_indices[0, m] = reverse_mapping[subgraph_regulation_edge_index[0, m].item()]  # Source node
                         old_edge_indices[1, m] = reverse_mapping[subgraph_regulation_edge_index[1, m].item()]

                    # if len(subgraph_regulation_edge_weight)==0:
                    #     continue
                    label_graph = Data(edge_index=subgraph_regulation_edge_index, edge_weight=subgraph_regulation_edge_weight)
                    #print(label_graph)
                    # Assign the label graph to the subgraph's y attribute
                    subgraph_data.y = label_graph.edge_index
                    self.all_sampled_indices.append(subgraph_node_set)
                    self.all_edges.append(old_edge_indices)
                   #node_attr =  torch.column_stack((subgraph_Is_TF,minmax_exp_acrosscell,))
                    self.data_list.append((subgraph_data,node_idx_map))


            
                first_hop_limit = int(self.max_subgraph_size/2)
                # Select up to 50 first-hop neighbors
                selected_first_hop = first_hop_neighbors[:first_hop_limit]

                # Calculate remaining slots
                remaining_slots = max_nodes - (len(selected_first_hop) + 1)

                # Select remaining nodes from second-hop neighbors
                selected_second_hop = second_hop_neighbors[:remaining_slots]

                # Combine tf_idx, first-hop, and second-hop neighbors
                subgraph_node_idx = torch.cat([torch.tensor([tf_idx]), selected_first_hop, selected_second_hop])
                subgraph_node_set = set(subgraph_node_idx.tolist())

                # Remap the subgraph node indices
                subgraph_node_idx = subgraph_node_idx.unique()
                node_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph_node_idx.tolist())}
                reverse_mapping = {v: k for k, v in node_idx_map.items()}

                # Filter edges to keep only those within the subgraph nodes
                mask = torch.isin(subgraph_edge_index1[0], subgraph_node_idx) & torch.isin(subgraph_edge_index1[1], subgraph_node_idx)
                subgraph_edge_index_filtered = subgraph_edge_index1[:, mask]
                subgraph_edge_index = torch.tensor([[node_idx_map[i.item()], node_idx_map[j.item()]] 
                                                    for i, j in zip(subgraph_edge_index_filtered[0], subgraph_edge_index_filtered[1])], 
                                                   dtype=torch.long).t().contiguous()

                # Get the corresponding edge weights
                subgraph_edge_weight = full_graph.edge_weight[
                    torch.isin(full_graph.edge_index[0], subgraph_node_idx) & 
                    torch.isin(full_graph.edge_index[1], subgraph_node_idx)
                ]

                # Create node features
                exp_x = full_graph.x[subgraph_node_idx]
                zscore = self.z_score_per_cell(exp_x)
                
                # Apply perturbations after normalization if requested
                if self.perturb_after_normalization and (self.noise_std > 0.0 or self.dropout_fraction > 0.0):
                    zscore_np = zscore.numpy() if isinstance(zscore, torch.Tensor) else zscore
                    if self.dropout_fraction > 0.0:
                        zscore_np = self.apply_dropout(zscore_np, self.dropout_fraction)
                    if self.noise_std > 0.0:
                        # Use noise_std as absolute value (normalized data typically has std ~1)
                        zscore_np = self.add_gaussian_noise(zscore_np, self.noise_std)
                    zscore = torch.tensor(zscore_np, dtype=torch.float32) if isinstance(zscore, torch.Tensor) else zscore_np
                
                subgraph_x = torch.column_stack((zscore, Is_tf[subgraph_node_idx]))
                print(subgraph_x.shape)
                # Create the subgraph
                subgraph_data = Data(x=subgraph_x, edge_index=subgraph_edge_index, edge_weight=subgraph_edge_weight)
                edge_weight1 = torch.cat([subgraph_data.edge_weight.unsqueeze(1)])

                subgraph_data.edge_weight = edge_weight1
                # Process the regulation matrix for the subgraph
                subgraph_indices = subgraph_node_idx.tolist()
                subgraph_regulation_matrix = regulation_matrix[:, subgraph_indices][subgraph_indices, :]
                subgraph_regulation_edge_index, subgraph_regulation_edge_weight = dense_to_sparse(torch.tensor(subgraph_regulation_matrix))

                # Remap regulation edge indices
                old_edge_indices = torch.zeros_like(subgraph_regulation_edge_index)#.cuda()
                for m in range(subgraph_regulation_edge_index.size(1)):  # Iterate over all edges
                    old_edge_indices[0, m] = reverse_mapping[subgraph_regulation_edge_index[0, m].item()]  # Source node
                    old_edge_indices[1, m] = reverse_mapping[subgraph_regulation_edge_index[1, m].item()]

                #if len(subgraph_regulation_edge_weight) == 0:
                #    continue
                                # After finalizing subgraph_node_idx
                if len(subgraph_node_idx) < 3:
                    print(f"Skipping degenerate subgraph with {len(subgraph_node_idx)} nodes (TF branch)")
                    continue
                # Create label graph for the subgraph
                label_graph = Data(edge_index=subgraph_regulation_edge_index, edge_weight=subgraph_regulation_edge_weight)
                subgraph_data.y = label_graph.edge_index

                # Append results
                self.all_sampled_indices.append(subgraph_node_set)
                self.all_edges.append(old_edge_indices)
                self.data_list.append((subgraph_data, node_idx_map))



        all_samples = set(chain.from_iterable(self.all_sampled_indices))
        all_nodes = set(range(full_graph.num_nodes))
        no_sampled_nodes = list(all_nodes - all_samples)
        print(no_sampled_nodes)
        print("sampling remaining nodes")
        # Step 3: For each remaining node, create a new subgraph centered on that node
        
        for rem_node in no_sampled_nodes:
                num_hops = 1
                while num_hops<10:
                    subgraph_node_idx, subgraph_edge_index, _, _ = k_hop_subgraph(rem_node, num_hops=num_hops, edge_index=full_graph.edge_index, num_nodes=num_genes)
                    if len(subgraph_node_idx) >= self.max_subgraph_size:
                        break
                    print(num_hops,len(subgraph_node_idx))
                    num_hops += 1
                
                #if len(subgraph_node_idx) < 500:
                #    continue
                #subgraph_node_idx, subgraph_edge_index, _, _ = k_hop_subgraph(tf_idx, num_hops=2, edge_index=full_graph.edge_index, num_nodes=num_genes)
                #print(subgraph_edge_index)
                first_hop_neighbors = full_graph.edge_index[1][full_graph.edge_index[0] == rem_node]
                first_hop_neighbors = first_hop_neighbors.unique()
                #print(len(first_hop_neighbors))
                first_hop_neighbors = first_hop_neighbors[first_hop_neighbors!= rem_node]
                #print(len(first_hop_neighbors))
                second_hop_neighbors = subgraph_node_idx[~torch.isin(subgraph_node_idx,first_hop_neighbors) & (subgraph_node_idx != rem_node)]
                
                # Combine TF, first hop, and second hop neighbors to get exactly 100 nodes
                if len(first_hop_neighbors) + 1 >= self.max_subgraph_size:
                    subgraph_node_idx = torch.cat([torch.tensor([rem_node]), first_hop_neighbors[:self.max_subgraph_size-1]])
                else:
                    remaining_nodes = self.max_subgraph_size - (len(first_hop_neighbors) + 1)
                    subgraph_node_idx = torch.cat([torch.tensor([rem_node]), first_hop_neighbors, second_hop_neighbors[:remaining_nodes]])
                
                subgraph_node_set = set(subgraph_node_idx.tolist())
                
                #print(torch.isin(rem_node,subgraph_node_idx))
                # Remap the subgraph node indices
                subgraph_node_idx = subgraph_node_idx.unique()
                #print(subgraph_node_idx)
                #print(len(subgraph_node_idx))
                node_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph_node_idx.tolist())}
                reverse_mapping = {v: k for k, v in node_idx_map.items()}
                # Filter edges to keep only those within the subgraph nodes
                mask = torch.isin(subgraph_edge_index[0], subgraph_node_idx) & torch.isin(subgraph_edge_index[1], subgraph_node_idx)
                subgraph_edge_index = subgraph_edge_index[:, mask]
                subgraph_edge_index = torch.tensor([[node_idx_map[i.item()], node_idx_map[j.item()]] for i, j in zip(subgraph_edge_index[0], subgraph_edge_index[1])], dtype=torch.long).t().contiguous()
                # Get the corresponding edge weights for the filtered edges
                subgraph_edge_weight = full_graph.edge_weight[
                    torch.isin(full_graph.edge_index[0], subgraph_node_idx) & 
                    torch.isin(full_graph.edge_index[1], subgraph_node_idx)
                ]
                #print(subgraph_edge_index)
                exp_x = full_graph.x[subgraph_node_idx]
                #minm_x = self.min_max(exp_x)
                zscore = self.z_score_per_cell(exp_x)
                
                # Apply perturbations after normalization if requested
                if self.perturb_after_normalization and (self.noise_std > 0.0 or self.dropout_fraction > 0.0):
                    zscore_np = zscore.numpy() if isinstance(zscore, torch.Tensor) else zscore
                    if self.dropout_fraction > 0.0:
                        zscore_np = self.apply_dropout(zscore_np, self.dropout_fraction)
                    if self.noise_std > 0.0:
                        # Use noise_std as absolute value (normalized data typically has std ~1)
                        zscore_np = self.add_gaussian_noise(zscore_np, self.noise_std)
                    zscore = torch.tensor(zscore_np, dtype=torch.float32) if isinstance(zscore, torch.Tensor) else zscore_np
                
                #print("zscore shape",zscore.shape)
                subgraph_x = torch.column_stack((zscore,Is_tf[subgraph_node_idx]))
                
                #subgraph_x = full_graph.x[subgraph_node_idx]
                
                subgraph_data = Data(x=subgraph_x, edge_index=subgraph_edge_index, edge_weight=subgraph_edge_weight)
                # #print(subgraph_data.edge_weight.shape,common_neigh[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1).shape)
                edge_weight1 = torch.cat([subgraph_data.edge_weight.unsqueeze(1)])

                subgraph_data.edge_weight = edge_weight1
 

                subgraph_indices = subgraph_node_idx.tolist()
                subgraph_regulation_matrix = regulation_matrix[:,subgraph_indices] [subgraph_indices,:]
                #print(subgraph_regulation_matrix)
                subgraph_regulation_edge_index, subgraph_regulation_edge_weight = dense_to_sparse(torch.tensor(subgraph_regulation_matrix))
                # print(old_edge_indices)
                #if len(subgraph_regulation_edge_weight)==0:
                #    continue
                label_graph = Data(edge_index=subgraph_regulation_edge_index, edge_weight=subgraph_regulation_edge_weight)
                if len(subgraph_node_idx) < 3:
                    print(f"Skipping degenerate subgraph with {len(subgraph_node_idx)} nodes (remaining node branch)")
                    continue
                #print(label_graph)
                # Assign the label graph to the subgraph's y attribute
                subgraph_data.y = label_graph.edge_index
                self.all_sampled_indices.append(subgraph_node_set)
                self.all_edges.append(old_edge_indices)
                self.data_list.append((subgraph_data,node_idx_map))


        
        #print(len(unique_edges))
        target_size = self.max_subgraph_size
        all_nodes = list(range(full_graph.num_nodes))
        #subgraphs = []

        target_per_node = target_size // 2  # Attempt to balance 50/50 between two nodes
        
       
        
        all_samples_final = set(chain.from_iterable(self.all_sampled_indices))
        print(all_samples_final)
        all_edges_flat = torch.cat(self.all_edges, dim=1)
        unique_edges = set(map(tuple, all_edges_flat.t().tolist()))
        
        print(len(unique_edges))
        
        torch.save(self.data_list, self.processed_paths[0])

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        subgraph_data,node_map=self.data_list[idx]
        data = subgraph_data.x
          
          
        edges = subgraph_data.edge_index
        #edge_weights = self.data_list[idx].edge_weight
        edge_weights = subgraph_data.edge_weight[:,0].unsqueeze(1)
        pos_edges = subgraph_data.y
        
        #gsum = self.SubaGraphs[idx].gsum
        #item = next(GCENdataset)
        return [data,edges,edge_weights, pos_edges,node_map]
