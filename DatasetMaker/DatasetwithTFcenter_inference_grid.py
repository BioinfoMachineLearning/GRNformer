import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.utils import from_networkx
import networkx as nx
from scipy.spatial.distance import pdist, squareform
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
    def __init__(self, root, gene_expression_file, tf_genes, regulation_file, transform=None, pre_transform=None):
        self.gene_expression_file = gene_expression_file
        self.tf_genes = tf_genes
        self.regulation_file = regulation_file
        super(GeneExpressionDataset, self).__init__(root, transform, pre_transform)
        self.data_list = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return [self.gene_expression_file, self.regulation_file]

    @property
    def processed_file_names(self):
        return [os.path.splitext(os.path.basename(self.regulation_file))[0]+'-inference_grid_updatedhop.pt']

    def download(self):
        pass
    def istf(self,genes,TFs):
        #Read human TFs list,and add a bool feature if a gene is a TF or not
        

        is_tf = genes.str.upper().isin(TFs)  
        #print(is_tf)
        return is_tf 
    def construct_networkx(self,edge_weight_matrix,gene_exp_matr):
            """
              Constructs a network from an edge weight matrix.

              Args:
                edge_weight_matrix: A square matrix of edge weights.

              Returns:
                A `torch_geometric.data.Data` object representing the network.
            """

            edge_index = (abs(edge_weight_matrix) > 0.1).nonzero().t()
            row, col = edge_index
            edge_weight = edge_weight_matrix[row, col]
            #G = nx.Graph(np.matrix(edge_weight_matrix))


            # Create a Data object to represent the graph.
            data = Data(x=torch.tensor(gene_exp_matr),edge_index=edge_index, edge_weight=edge_weight)


              # Return the data object.
            return data    

    def common_neighbors(self,edge_index, num_nodes):
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        return torch.mm(adj, adj)

    # Function to calculate Jaccard coefficient
    def jaccard_coefficient(self,edge_index, num_nodes):
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        intersection = torch.mm(adj, adj)
        row_sums = adj.sum(dim=1)
        denominator = row_sums.unsqueeze(0) + row_sums.unsqueeze(1) - intersection
        return intersection.float() / denominator.float()

    # Function to calculate Adamic/Adar index
    def adamic_adar_index(self,edge_index, num_nodes):
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        degrees = adj.sum(dim=1).float()
        degrees[degrees == 0] = float('inf')  # Prevent division by zero
        inv_log_degrees = 1 / torch.log(degrees)
        return torch.mm(adj * inv_log_degrees.unsqueeze(0), adj.t())

    # Function to calculate preferential attachment
    def preferential_attachment(self,edge_index, num_nodes):
        degrees = degree(edge_index[0], num_nodes=num_nodes)
        return degrees.unsqueeze(1) * degrees.unsqueeze(0)
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

    
    def process(self):
        
        self.data_list = []
        self.all_sampled_indices=[]
        self.all_edges=[]
        self.node_idx_maps=[]
        # Step 1: Create a weighted gene co-expression network
        gene_expression_data = pd.read_csv(self.raw_paths[0], index_col=0)
        gene_expression_data = gene_expression_data.apply(np.arcsinh)
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
                
        #print(len(regulation_data),(tf_gene in regulation_data))
        regulation_data['Type']=1
        #print(regulation_data)
        regulation_matrix = np.zeros((num_genes, num_genes))
        genes = gene_expression_data.index
        Exp_tar_adj = pd.DataFrame(index=gene_indices.keys(), columns=gene_indices.keys())
        # Fill in the values based on df2
        for index, row in regulation_data.iterrows():
            if(row['Gene1'] in genes)and (row['Gene2'] in genes):
                Exp_tar_adj.at[row['Gene1'], row['Gene2']] = row['Type']
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
                while num_hops<100:
                    subgraph_node_idx1, subgraph_edge_index1, _, _ = k_hop_subgraph(tf_idx, num_hops=num_hops, edge_index=full_graph.edge_index, num_nodes=num_genes)
                    if len(subgraph_node_idx1) >= 100:
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
                # Combine TF, first hop, and second hop neighbors to get exactly 100 nodes
                #cover all the first hop neighbors of TF with each subgraph with 99 neighbors
                


                # if len(first_hop_neighbors) + 1 >= 100:
                #     subgraph_node_idx = torch.cat([torch.tensor([tf_idx]), first_hop_neighbors[:99]])
                # else:
                #     remaining_nodes = 100 - (len(first_hop_neighbors) + 1)
                #     subgraph_node_idx = torch.cat([torch.tensor([tf_idx]), first_hop_neighbors, second_hop_neighbors[:remaining_nodes]])
                max_nodes = 100
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
                    #print("zscore shape",zscore.shape)
                    subgraph_x = torch.column_stack((zscore,Is_tf[subgraph_node_idx]))

                    #subgraph_x = full_graph.x[subgraph_node_idx]

                    subgraph_data = Data(x=subgraph_x, edge_index=subgraph_edge_index, edge_weight=subgraph_edge_weight)
                    #print("subgraph:",subgraph_data)
                    # Step 3: Create unique label graph for the subgraph
                    # common_neigh = self.common_neighbors(subgraph_data.edge_index, len(subgraph_node_idx))

                    # jaccard = self.jaccard_coefficient(subgraph_data.edge_index, len(subgraph_node_idx))

                    # adamic_adar = self.adamic_adar_index(subgraph_data.edge_index, len(subgraph_node_idx))
                    # preferential_attach = self.preferential_attachment(subgraph_data.edge_index, len(subgraph_node_idx))
                    # #print(subgraph_data.edge_weight.shape,common_neigh[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1).shape)
                    edge_weight1 = torch.cat([subgraph_data.edge_weight.unsqueeze(1)])
                    #             common_neigh[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1).float(),
                    #             jaccard[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1),
                    #             adamic_adar[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1),
                    #             preferential_attach[subgraph_data.edge_index[0],subgraph_data.edge_index[1]].unsqueeze(1)], dim=1)

                    subgraph_data.edge_weight = edge_weight1
    
                    #for _, row in regulation_data.iterrows():
                    #    gene1, gene2, regulation_type = row
                    #    if gene1 in gene_indices and gene2 in gene_indices:
                    #        idx1 = gene_indices[gene1]
                    #        idx2 = gene_indices[gene2]
                    #        regulation_matrix[idx1, idx2] = regulation_type

                    subgraph_indices = subgraph_node_idx.tolist()
                    subgraph_regulation_matrix = regulation_matrix[:,subgraph_indices] [subgraph_indices,:]
                    #print(subgraph_regulation_matrix,regulation_matrix.sum())
                    subgraph_regulation_edge_index, subgraph_regulation_edge_weight = dense_to_sparse(torch.tensor(subgraph_regulation_matrix))
                    old_edge_indices = torch.zeros_like(subgraph_regulation_edge_index).cuda()
                    for m in range(subgraph_regulation_edge_index.size(1)):  # Iterate over all edges
                         old_edge_indices[0, m] = reverse_mapping[subgraph_regulation_edge_index[0, m].item()]  # Source node
                         old_edge_indices[1, m] = reverse_mapping[subgraph_regulation_edge_index[1, m].item()]

                    if len(subgraph_regulation_edge_weight)==0:
                        continue
                    label_graph = Data(edge_index=subgraph_regulation_edge_index, edge_weight=subgraph_regulation_edge_weight)
                    #print(label_graph)
                    # Assign the label graph to the subgraph's y attribute
                    subgraph_data.y = label_graph.edge_index
                    self.all_sampled_indices.append(subgraph_node_set)
                    self.all_edges.append(old_edge_indices)
                    #self.node_idx_maps.append(node_idx_map)
                    #subgraph_Is_TF =Is_tf[subgraph_node_idx]
                    #Z_exp_acrossgenes = self.z_score(subgraph_x.t())

                    #Z_exp_acrosscell = self.z_score_dim1(subgraph_x)
                    #minmax_exp_acrosscell = self.min_max_dim0(Z_exp_acrosscell)
                    #print(minm_acrossgenes.shape,minmax_exp_acrosscell.shape)
                    #exp_mat_df = torch.as_tensor(Exp_data_df.to_numpy())


                    #node_attr =  torch.column_stack((subgraph_Is_TF,minmax_exp_acrosscell,))
                    self.data_list.append((subgraph_data,node_idx_map))


            
                first_hop_limit = 50
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
                subgraph_x = torch.column_stack((zscore, Is_tf[subgraph_node_idx]))

                # Create the subgraph
                subgraph_data = Data(x=subgraph_x, edge_index=subgraph_edge_index, edge_weight=subgraph_edge_weight)
                edge_weight1 = torch.cat([subgraph_data.edge_weight.unsqueeze(1)])
                    #             common_neigh[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1).float(),
                    #             jaccard[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1),
                    #             adamic_adar[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1),
                    #             preferential_attach[subgraph_data.edge_index[0],subgraph_data.edge_index[1]].unsqueeze(1)], dim=1)

                subgraph_data.edge_weight = edge_weight1
                # Process the regulation matrix for the subgraph
                subgraph_indices = subgraph_node_idx.tolist()
                subgraph_regulation_matrix = regulation_matrix[:, subgraph_indices][subgraph_indices, :]
                subgraph_regulation_edge_index, subgraph_regulation_edge_weight = dense_to_sparse(torch.tensor(subgraph_regulation_matrix))

                # Remap regulation edge indices
                old_edge_indices = torch.zeros_like(subgraph_regulation_edge_index).cuda()
                for m in range(subgraph_regulation_edge_index.size(1)):  # Iterate over all edges
                    old_edge_indices[0, m] = reverse_mapping[subgraph_regulation_edge_index[0, m].item()]  # Source node
                    old_edge_indices[1, m] = reverse_mapping[subgraph_regulation_edge_index[1, m].item()]

                if len(subgraph_regulation_edge_weight) == 0:
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
                while num_hops<100:
                    subgraph_node_idx, subgraph_edge_index, _, _ = k_hop_subgraph(rem_node, num_hops=num_hops, edge_index=full_graph.edge_index, num_nodes=num_genes)
                    if len(subgraph_node_idx) >= 100:
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
                if len(first_hop_neighbors) + 1 >= 100:
                    subgraph_node_idx = torch.cat([torch.tensor([rem_node]), first_hop_neighbors[:99]])
                else:
                    remaining_nodes = 100 - (len(first_hop_neighbors) + 1)
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
                #print("zscore shape",zscore.shape)
                subgraph_x = torch.column_stack((zscore,Is_tf[subgraph_node_idx]))
                
                #subgraph_x = full_graph.x[subgraph_node_idx]
                
                subgraph_data = Data(x=subgraph_x, edge_index=subgraph_edge_index, edge_weight=subgraph_edge_weight)
                #print("subgraph:",subgraph_data)
                # Step 3: Create unique label graph for the subgraph
                # common_neigh = self.common_neighbors(subgraph_data.edge_index, len(subgraph_node_idx))

                # jaccard = self.jaccard_coefficient(subgraph_data.edge_index, len(subgraph_node_idx))

                # adamic_adar = self.adamic_adar_index(subgraph_data.edge_index, len(subgraph_node_idx))
                # preferential_attach = self.preferential_attachment(subgraph_data.edge_index, len(subgraph_node_idx))
                # #print(subgraph_data.edge_weight.shape,common_neigh[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1).shape)
                edge_weight1 = torch.cat([subgraph_data.edge_weight.unsqueeze(1)])
                #             common_neigh[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1).float(),
                #             jaccard[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1),
                #             adamic_adar[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1),
                #             preferential_attach[subgraph_data.edge_index[0],subgraph_data.edge_index[1]].unsqueeze(1)], dim=1)

                subgraph_data.edge_weight = edge_weight1
 
                #for _, row in regulation_data.iterrows():
                #    gene1, gene2, regulation_type = row
                #    if gene1 in gene_indices and gene2 in gene_indices:
                #        idx1 = gene_indices[gene1]
                #        idx2 = gene_indices[gene2]
                #        regulation_matrix[idx1, idx2] = regulation_type

                subgraph_indices = subgraph_node_idx.tolist()
                subgraph_regulation_matrix = regulation_matrix[:,subgraph_indices] [subgraph_indices,:]
                #print(subgraph_regulation_matrix)
                subgraph_regulation_edge_index, subgraph_regulation_edge_weight = dense_to_sparse(torch.tensor(subgraph_regulation_matrix))
                #print(subgraph_regulation_edge_index)
                # old_edge_indices = torch.zeros_like(subgraph_regulation_edge_index).cuda()
                # for m in range(subgraph_regulation_edge_index.size(1)):  # Iterate over all edges
                #     old_edge_indices[0, m] = reverse_mapping[subgraph_regulation_edge_index[0, m].item()]  # Source node
                #     old_edge_indices[1, m] = reverse_mapping[subgraph_regulation_edge_index[1, m].item()]
                # print(old_edge_indices)
                if len(subgraph_regulation_edge_weight)==0:
                    continue
                label_graph = Data(edge_index=subgraph_regulation_edge_index, edge_weight=subgraph_regulation_edge_weight)
                #print(label_graph)
                # Assign the label graph to the subgraph's y attribute
                subgraph_data.y = label_graph.edge_index
                self.all_sampled_indices.append(subgraph_node_set)
                self.all_edges.append(old_edge_indices)
                #self.node_idx_maps.append(node_idx_map)
                #subgraph_Is_TF =Is_tf[subgraph_node_idx]
                #Z_exp_acrossgenes = self.z_score(subgraph_x.t())
                
                #Z_exp_acrosscell = self.z_score_dim1(subgraph_x)
                #minmax_exp_acrosscell = self.min_max_dim0(Z_exp_acrosscell)
                #print(minm_acrossgenes.shape,minmax_exp_acrosscell.shape)
                #exp_mat_df = torch.as_tensor(Exp_data_df.to_numpy())
                

                #node_attr =  torch.column_stack((subgraph_Is_TF,minmax_exp_acrosscell,))
                self.data_list.append((subgraph_data,node_idx_map))
        
        # Step 4: Save the processed data
        #data, slices = self.collate(data_list)
        #all_samples_final = set(chain.from_iterable(self.all_sampled_indices))
        #print(all_samples_final)
        #all_edges_flat = torch.cat(self.all_edges, dim=1)
        #unique_edges = set(map(tuple, all_edges_flat.t().tolist()))
        
        #print(len(unique_edges))
        num_subgraphs=100
        all_nodes = list(range(full_graph.num_nodes))
        #subgraphs = []
        '''
        for _ in range(num_subgraphs):
            # Randomly sample 'subgraph_size' nodes
            subgraph_nodes = random.sample(all_nodes, 100)
            
                
            #print(torch.isin(rem_node,subgraph_node_idx))
            # Remap the subgraph node indices
            subgraph_node_idx = torch.as_tensor(subgraph_nodes)
            subgraph_node_set = set(subgraph_node_idx.tolist())
            #print(subgraph_node_idx)
            #print(len(subgraph_node_idx))
            node_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph_node_idx.tolist())}
            reverse_mapping = {v: k for k, v in node_idx_map.items()}
            # Filter edges to keep only those within the subgraph nodes
            
            mask = torch.isin(full_graph.edge_index[0], subgraph_node_idx) & torch.isin(full_graph.edge_index[1], subgraph_node_idx)
            subgraph_edge_index = full_graph.edge_index[:, mask]
            subgraph_edge_index = torch.tensor([[node_idx_map[i.item()], node_idx_map[j.item()]] for i, j in zip(subgraph_edge_index[0], subgraph_edge_index[1])], dtype=torch.long).t().contiguous()
            #if subgraph_edge_index.size(1)==0:
            #    continue
            # Get the corresponding edge weights for the filtered edges
            subgraph_edge_weight = full_graph.edge_weight[
                torch.isin(full_graph.edge_index[0], subgraph_node_idx) & 
                torch.isin(full_graph.edge_index[1], subgraph_node_idx)
            ]
            #print(subgraph_edge_index)
            exp_x = full_graph.x[subgraph_node_idx]
            #minm_x = self.min_max(exp_x)
            zscore = self.z_score_per_cell(exp_x)
            #print("zscore shape",zscore.shape)
            subgraph_x = torch.column_stack((zscore,Is_tf[subgraph_node_idx]))
            
            #subgraph_x = full_graph.x[subgraph_node_idx]
            
            subgraph_data = Data(x=subgraph_x, edge_index=subgraph_edge_index, edge_weight=subgraph_edge_weight)
            #print("subgraph:",subgraph_data)
            # Step 3: Create unique label graph for the subgraph
            #common_neigh = self.common_neighbors(subgraph_data.edge_index, len(subgraph_node_idx))
            #jaccard = self.jaccard_coefficient(subgraph_data.edge_index, len(subgraph_node_idx))
            #adamic_adar = self.adamic_adar_index(subgraph_data.edge_index, len(subgraph_node_idx))
            #preferential_attach = self.preferential_attachment(subgraph_data.edge_index, len(subgraph_node_idx))
            #print(subgraph_data.edge_weight.shape,common_neigh[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1).shape)
            edge_weight1 = torch.cat([subgraph_data.edge_weight.unsqueeze(1)])
            #            common_neigh[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1).float(),
            #            jaccard[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1),
            #            adamic_adar[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1),
            #           preferential_attach[subgraph_data.edge_index[0],subgraph_data.edge_index[1]].unsqueeze(1)], dim=1)
            subgraph_data.edge_weight = edge_weight1
            subgraph_indices = subgraph_node_idx.tolist()
            subgraph_regulation_matrix = regulation_matrix[:,subgraph_indices] [subgraph_indices,:]
            #print(subgraph_regulation_matrix)
            subgraph_regulation_edge_index, subgraph_regulation_edge_weight = dense_to_sparse(torch.tensor(subgraph_regulation_matrix))
            #print(subgraph_regulation_edge_index)
            old_edge_indices = torch.zeros_like(subgraph_regulation_edge_index).cuda()
            for m in range(subgraph_regulation_edge_index.size(1)):  # Iterate over all edges
                old_edge_indices[0, m] = reverse_mapping[subgraph_regulation_edge_index[0, m].item()]  # Source node
                old_edge_indices[1, m] = reverse_mapping[subgraph_regulation_edge_index[1, m].item()]
            print(old_edge_indices)
            if len(subgraph_regulation_edge_weight)==0:
                continue
            label_graph = Data(edge_index=subgraph_regulation_edge_index, edge_weight=subgraph_regulation_edge_weight)
            #print(label_graph)
            # Assign the label graph to the subgraph's y attribute
            subgraph_data.y = label_graph.edge_index
            self.all_sampled_indices.append(subgraph_node_set)
            self.all_edges.append(old_edge_indices)
            #self.node_idx_maps.append(node_idx_map)
            #subgraph_Is_TF =Is_tf[subgraph_node_idx]
            #Z_exp_acrossgenes = self.z_score(subgraph_x.t())
            
            #Z_exp_acrosscell = self.z_score_dim1(subgraph_x)
            #minmax_exp_acrosscell = self.min_max_dim0(Z_exp_acrosscell)
            #print(minm_acrossgenes.shape,minmax_exp_acrosscell.shape)
            #exp_mat_df = torch.as_tensor(Exp_data_df.to_numpy())
            
            #node_attr =  torch.column_stack((subgraph_Is_TF,minmax_exp_acrosscell,))
            self.data_list.append((subgraph_data,node_idx_map))
        '''
        target_size=100
        target_per_node = target_size // 2  # Attempt to balance 50/50 between two nodes
        
        # # Step 1: Randomly select two distinct nodes
        # for _ in range(num_subgraphs):
        #     node1, node2 = random.sample(all_nodes, 2)
        #     hop = 1
        #     subgraph_nodes = set()
        #     # Step 2: Incrementally expand neighbors until we reach the target size
        #     while len(subgraph_nodes) < target_size:
        #         # Get hop neighbors for each node
        #         nodes1, edge_index1, _, _ = k_hop_subgraph(node1, hop, full_graph.edge_index, relabel_nodes=False)
        #         nodes2, edge_index2, _, _ = k_hop_subgraph(node2, hop, full_graph.edge_index, relabel_nodes=False)
        #         # Combine neighbors into a unique set
        #         combined_neighbors = set(nodes1.tolist()).union(set(nodes2.tolist()))
        #         # Check if we reached or exceeded the target
        #         if len(combined_neighbors) >= target_size:
        #             break
        #         hop += 1  # Increase hop count and try again
        #     # Step 3: Balance neighbors to get as close to 50/50 as possible
        #     neighbors1 = list(set(nodes1.tolist()))
        #     neighbors2 = list(set(nodes2.tolist()))
        #     # Shuffle to introduce randomness and prevent bias
        #     random.shuffle(neighbors1)
        #     random.shuffle(neighbors2)
        #     # Select up to 49 nodes from each set
        #     balanced_neighbors1 = neighbors1[:min(target_per_node, len(neighbors1))]
        #     balanced_neighbors2 = neighbors2[:min(target_per_node, len(neighbors2))]
        #     # Merge neighbors and adjust for any remaining required nodes
        #     final_neighbors = set(balanced_neighbors1).union(set(balanced_neighbors2))
        #     # If the combined set is still smaller than the target, increase hop count
        #     if len(final_neighbors) < target_size:
        #         remaining_needed = target_size - len(final_neighbors)
        #         additional_neighbors = list(combined_neighbors - final_neighbors)[:remaining_needed]
        #         final_neighbors.update(additional_neighbors)
        #     # Step 4: Convert final set to a tensor and retrieve subgraph edges and features
        #     subgraph_node_idx = torch.tensor(list(final_neighbors), dtype=torch.long)
        #     subgraph_node_set = set(subgraph_node_idx.tolist())
        #     #print(subgraph_node_idx)
        #     #print(len(subgraph_node_idx))
        #     node_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph_node_idx.tolist())}
        #     reverse_mapping = {v: k for k, v in node_idx_map.items()}
        #     # Filter edges to keep only those within the subgraph nodes
        #     mask = torch.isin(full_graph.edge_index[0], subgraph_node_idx) & torch.isin(full_graph.edge_index[1], subgraph_node_idx)
        #     subgraph_edge_index = full_graph.edge_index[:, mask]
        #     subgraph_edge_index = torch.tensor([[node_idx_map[i.item()], node_idx_map[j.item()]] for i, j in zip(subgraph_edge_index[0], subgraph_edge_index[1])], dtype=torch.long).t().contiguous()
        #     #if subgraph_edge_index.size(1)==0:
        #     #    continue
        #     # Get the corresponding edge weights for the filtered edges
        #     subgraph_edge_weight = full_graph.edge_weight[
        #         torch.isin(full_graph.edge_index[0], subgraph_node_idx) & 
        #         torch.isin(full_graph.edge_index[1], subgraph_node_idx)
        #     ]
        #     #print(subgraph_edge_index)
        #     exp_x = full_graph.x[subgraph_node_idx]
        #     #minm_x = self.min_max(exp_x)
        #     zscore = self.z_score_per_cell(exp_x)
        #     #print("zscore shape",zscore.shape)
        #     subgraph_x = torch.column_stack((zscore,Is_tf[subgraph_node_idx]))
        #     #subgraph_x = full_graph.x[subgraph_node_idx]
        #     subgraph_data = Data(x=subgraph_x, edge_index=subgraph_edge_index, edge_weight=subgraph_edge_weight)
        #     #print("subgraph:",subgraph_data)
        #     # Step 3: Create unique label graph for the subgraph
        #     #common_neigh = self.common_neighbors(subgraph_data.edge_index, len(subgraph_node_idx))
        #     #jaccard = self.jaccard_coefficient(subgraph_data.edge_index, len(subgraph_node_idx))
        #     #adamic_adar = self.adamic_adar_index(subgraph_data.edge_index, len(subgraph_node_idx))
        #     #preferential_attach = self.preferential_attachment(subgraph_data.edge_index, len(subgraph_node_idx))
        #     #print(subgraph_data.edge_weight.shape,common_neigh[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1).shape)
        #     edge_weight1 = torch.cat([subgraph_data.edge_weight.unsqueeze(1)])
        #     #            common_neigh[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1).float(),
        #     #            jaccard[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1),
        #     #            adamic_adar[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1),
        #     #           preferential_attach[subgraph_data.edge_index[0],subgraph_data.edge_index[1]].unsqueeze(1)], dim=1)
        #     subgraph_data.edge_weight = edge_weight1
        #     subgraph_indices = subgraph_node_idx.tolist()
        #     subgraph_regulation_matrix = regulation_matrix[:,subgraph_indices] [subgraph_indices,:]
        #     #print(subgraph_regulation_matrix)
        #     subgraph_regulation_edge_index, subgraph_regulation_edge_weight = dense_to_sparse(torch.tensor(subgraph_regulation_matrix))
        #     #print(subgraph_regulation_edge_index)
        #     old_edge_indices = torch.zeros_like(subgraph_regulation_edge_index).cuda()
        #     for m in range(subgraph_regulation_edge_index.size(1)):  # Iterate over all edges
        #         old_edge_indices[0, m] = reverse_mapping[subgraph_regulation_edge_index[0, m].item()]  # Source node
        #         old_edge_indices[1, m] = reverse_mapping[subgraph_regulation_edge_index[1, m].item()]
        #     #print(old_edge_indices)
        #     if len(subgraph_regulation_edge_weight)==0:
        #         continue
        #     label_graph = Data(edge_index=subgraph_regulation_edge_index, edge_weight=subgraph_regulation_edge_weight)
        #     #print(label_graph)
        #     # Assign the label graph to the subgraph's y attribute
        #     subgraph_data.y = label_graph.edge_index
        #     self.all_sampled_indices.append(subgraph_node_set)
        #     self.all_edges.append(old_edge_indices)
        #     #self.node_idx_maps.append(node_idx_map)
        #     #subgraph_Is_TF =Is_tf[subgraph_node_idx]
        #     #Z_exp_acrossgenes = self.z_score(subgraph_x.t())
        #     #Z_exp_acrosscell = self.z_score_dim1(subgraph_x)
        #     #minmax_exp_acrosscell = self.min_max_dim0(Z_exp_acrosscell)
        #     #print(minm_acrossgenes.shape,minmax_exp_acrosscell.shape)
        #     #exp_mat_df = torch.as_tensor(Exp_data_df.to_numpy())
        #     #node_attr =  torch.column_stack((subgraph_Is_TF,minmax_exp_acrosscell,))
        #     self.data_list.append((subgraph_data,node_idx_map))
    
        
        
        
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
# Usage
'''
print("hi")
root = [os.path.abspath('Data/sc-RNA-seq/hESC'),os.path.abspath('Data/sc-RNA-seq/hHep'),
            os.path.abspath('Data/sc-RNA-seq/mDC'),os.path.abspath('Data/sc-RNA-seq/mHSC-E'),
            os.path.abspath('Data/sc-RNA-seq/mHSC-GM'),os.path.abspath('Data/sc-RNA-seq/mESC'),
                            os.path.abspath('Data/sc-RNA-seq/mHSC-L')]

gene_expression_file = [os.path.abspath('Data/sc-RNA-seq/hESC/ExpressionData.csv'),os.path.abspath('Data/sc-RNA-seq/hHep/ExpressionData.csv'),
                            os.path.abspath('Data/sc-RNA-seq/mDC/ExpressionData.csv'),os.path.abspath('Data/sc-RNA-seq/mHSC-E/ExpressionData.csv'),
                            os.path.abspath('Data/sc-RNA-seq/mHSC-GM/ExpressionData.csv'),os.path.abspath('Data/sc-RNA-seq/mESC/ExpressionData.csv'),
                            os.path.abspath('Data/sc-RNA-seq/mHSC-L/ExpressionData.csv')]

tfhum_genes = pd.read_csv(os.path.abspath("Data/sc-RNA-seq/hESC/TFHumans.csv"),header=None)[0].to_list()
tfmou_genes = pd.read_csv(os.path.abspath("Data/sc-RNA-seq/mDC/TFMouse.csv"))['TF'].to_list()
TF_list = [tfhum_genes,tfhum_genes,tfmou_genes,tfmou_genes,tfmou_genes,tfmou_genes,tfmou_genes]
    # replace with actual TF gene names
regulation_file = [os.path.abspath('Data/sc-RNA-seq/hESC/hESC_combined.csv'),os.path.abspath('Data/sc-RNA-seq/hHep/hHep_combined.csv'),
                       os.path.abspath('Data/sc-RNA-seq/mDC/mDC_combined.csv'), os.path.abspath('Data/sc-RNA-seq/mHSC-E/mHSC-E_combined.csv'),
                        os.path.abspath('Data/sc-RNA-seq/mHSC-GM/mHSC-GM_combined.csv'),os.path.abspath('Data/sc-RNA-seq/mESC/mESC_combined.csv'),
                        os.path.abspath('Data/sc-RNA-seq/mHSC-L/mHSC-L_combined.csv')]
split_ind = [os.path.abspath("Data/sc-RNA-seq/hESC/dataset_splits_combined.pt"),os.path.abspath("Data/sc-RNA-seq/hHep/dataset_splits_combined.pt"),
                os.path.abspath("Data/sc-RNA-seq/mDC/dataset_splits_combined.pt") ,os.path.abspath("Data/sc-RNA-seq/mHSC-E/dataset_splits_combined.pt"),
                os.path.abspath("Data/sc-RNA-seq/mHSC-GM/dataset_splits_combined.pt"),os.path.abspath("Data/sc-RNA-seq/mESC/dataset_splits_combined.pt"),
                os.path.abspath("Data/sc-RNA-seq/mHSC-L/dataset_splits_combined.pt")]


for i in range(0,len(root)):

        dataset = GeneExpressionDataset(root[i],gene_expression_file[i],TF_list[i],regulation_file[i])

        print(len(dataset))
        train_size = int(0.70 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - (train_size+val_size)
        dataset_train,dataset_valid,dataset_test = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

        train_indices = dataset_train.indices
        valid_indices = dataset_valid.indices
        test_indices = dataset_test.indices
        torch.save({'train_indices': train_indices,'valid_indices':valid_indices, 'test_indices': test_indices}, split_ind[i])



root = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/hESC'
gene_expression_file = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/hESC/ExpressionData.csv'
tf_genes = pd.read_csv("C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/hESC/TFHumans.csv",header=None)[0].to_list()
# replace with actual TF gene names
regulation_file = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/hESC/hESC_combined.csv'

#dataset = torch.load("../Data/sc-RNA-seq/hHep/processed/data.pt")
dataset = GeneExpressionDataset(root,gene_expression_file,tf_genes,regulation_file)
#data_load =  DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=1)

# Iterate through the DataLoader to access individual data points
#print(len(data_load))
print(len(dataset))
train_size = int(0.70 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - (train_size+val_size)
dataset_train,dataset_valid,dataset_test = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

train_indices = dataset_train.indices
valid_indices = dataset_valid.indices
test_indices = dataset_test.indices
torch.save({'train_indices': train_indices,'valid_indices':valid_indices, 'test_indices': test_indices}, "C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/hESC/dataset_splits_combined.pt")


root = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/hHep'
gene_expression_file = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/hHep/ExpressionData.csv'
tf_genes = pd.read_csv("C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/hHep/TFHumans.csv",header=None)[0].to_list()
# replace with actual TF gene names
regulation_file = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/hHep/hHep_combined.csv'

#dataset = torch.load("../Data/sc-RNA-seq/hHep/processed/data.pt")
dataset = GeneExpressionDataset(root,gene_expression_file,tf_genes,regulation_file)
#data_load =  DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=1)

# Iterate through the DataLoader to access individual data points
#print(len(data_load))
print(len(dataset))
train_size = int(0.70 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - (train_size+val_size)
dataset_train,dataset_valid,dataset_test = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

train_indices = dataset_train.indices
valid_indices = dataset_valid.indices
test_indices = dataset_test.indices
torch.save({'train_indices': train_indices,'valid_indices':valid_indices, 'test_indices': test_indices}, "C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/hHep/dataset_splits_combined.pt")


#dataset = torch.load("../Data/sc-RNA-seq/hHep/processed/data.pt")
#dataset = GeneExpressionDataset(root,gene_expression_file,tf_genes,regulation_file)
#data_load =  DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=1)

# Iterate through the DataLoader to access individual data points
#print(len(data_load))
#for data in data_load:
#    print(data[0].shape)
root = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mDC'
gene_expression_file = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mDC/ExpressionData.csv'
tf_genes = pd.read_csv("C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mDC/TFMouse.csv")['TF'].to_list()
# replace with actual TF gene names
regulation_file = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mDC/mDC_combined.csv'

#dataset = torch.load("../Data/sc-RNA-seq/hHep/processed/data.pt")
dataset = GeneExpressionDataset(root,gene_expression_file,tf_genes,regulation_file)
#data_load =  DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=1)

# Iterate through the DataLoader to access individual data points
#print(len(data_load))
print(len(dataset))
train_size = int(0.70 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - (train_size+val_size)
dataset_train,dataset_valid,dataset_test = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

train_indices = dataset_train.indices
valid_indices = dataset_valid.indices
test_indices = dataset_test.indices
torch.save({'train_indices': train_indices,'valid_indices':valid_indices, 'test_indices': test_indices}, "C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mDC/dataset_splits_combined.pt")


root = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mESC'
gene_expression_file = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mESC/ExpressionData.csv'
tf_genes = pd.read_csv("C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mESC/TFMouse.csv")['TF'].to_list()
# replace with actual TF gene names
regulation_file = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mESC/mESC_combined.csv'

#dataset = torch.load("../Data/sc-RNA-seq/hHep/processed/data.pt")
dataset = GeneExpressionDataset(root,gene_expression_file,tf_genes,regulation_file)
#data_load =  DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=1)

# Iterate through the DataLoader to access individual data points
#print(len(data_load))
print(len(dataset))
train_size = int(0.70 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - (train_size+val_size)
dataset_train,dataset_valid,dataset_test = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

train_indices = dataset_train.indices
valid_indices = dataset_valid.indices
test_indices = dataset_test.indices
torch.save({'train_indices': train_indices,'valid_indices':valid_indices, 'test_indices': test_indices}, "C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mESC/dataset_splits_combined.pt")


root = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mHSC-E'
gene_expression_file = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mHSC-E/ExpressionData.csv'
tf_genes = pd.read_csv("C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mHSC-E/TFMouse.csv")['TF'].to_list()
# replace with actual TF gene names
regulation_file = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mHSC-E/mHSC-E_combined.csv'

#dataset = torch.load("../Data/sc-RNA-seq/hHep/processed/data.pt")
dataset = GeneExpressionDataset(root,gene_expression_file,tf_genes,regulation_file)
#data_load =  DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=1)

# Iterate through the DataLoader to access individual data points
#print(len(data_load))
print(len(dataset))
train_size = int(0.70 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - (train_size+val_size)
dataset_train,dataset_valid,dataset_test = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

train_indices = dataset_train.indices
valid_indices = dataset_valid.indices
test_indices = dataset_test.indices
torch.save({'train_indices': train_indices,'valid_indices':valid_indices, 'test_indices': test_indices}, "C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mHSC-E/dataset_splits_combined.pt")


root = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mHSC-GM'
gene_expression_file = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mHSC-GM/ExpressionData.csv'
tf_genes = pd.read_csv("C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mHSC-GM/TFMouse.csv")['TF'].to_list()
# replace with actual TF gene names
regulation_file = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mHSC-GM/mHSC-GM_combined.csv'

#dataset = torch.load("../Data/sc-RNA-seq/hHep/processed/data.pt")
dataset = GeneExpressionDataset(root,gene_expression_file,tf_genes,regulation_file)
#data_load =  DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=1)

# Iterate through the DataLoader to access individual data points
#print(len(data_load))
print(len(dataset))
train_size = int(0.70 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - (train_size+val_size)
dataset_train,dataset_valid,dataset_test = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

train_indices = dataset_train.indices
valid_indices = dataset_valid.indices
test_indices = dataset_test.indices
torch.save({'train_indices': train_indices,'valid_indices':valid_indices, 'test_indices': test_indices}, "C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mHSC-GM/dataset_splits_combined.pt")

root = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mHSC-L'
gene_expression_file = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mHSC-L/ExpressionData.csv'
tf_genes = pd.read_csv("C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mHSC-L/TFMouse.csv")['TF'].to_list()
# replace with actual TF gene names
regulation_file = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mHSC-L/mHSC-L_combined.csv'

#dataset = torch.load("../Data/sc-RNA-seq/hHep/processed/data.pt")
dataset = GeneExpressionDataset(root,gene_expression_file,tf_genes,regulation_file)
#data_load =  DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=1)

# Iterate through the DataLoader to access individual data points
#print(len(data_load))
print(len(dataset))
train_size = int(0.70 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - (train_size+val_size)
dataset_train,dataset_valid,dataset_test = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

train_indices = dataset_train.indices
valid_indices = dataset_valid.indices
test_indices = dataset_test.indices
torch.save({'train_indices': train_indices,'valid_indices':valid_indices, 'test_indices': test_indices}, "C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/mHSC-L/dataset_splits_combined.pt")
'''
