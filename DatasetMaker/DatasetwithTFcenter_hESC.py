import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.utils import from_networkx
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from torch_geometric.loader import NeighborSampler
from torch.utils.data import DataLoader
import os
"""
class GeneExpressionDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        #self.gene_expression_file = gene_expression_file
        self.gene_expression_file='ExpressionData.csv'
        self.tf_genes = pd.read_csv("/home/aghktb/GRN/GRNformer/Data/sc-RNA-seq/hHep/TFHumans.csv",header=None)[0].to_list()
        # replace with actual TF gene names
        self.regulation_file = 'HepG2-ChIP-seq-network.csv'
        #self.tf_genes = tf_genes
        #self.regulation_file = regulation_file
        super(GeneExpressionDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.gene_expression_file, self.regulation_file]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        # Step 1: Create a weighted gene co-expression network
        gene_expression_data = pd.read_csv(self.raw_paths[0], index_col=0)
        correlation_matrix = gene_expression_data.corr()
        edges = np.array(np.nonzero(correlation_matrix)).T
        weights = correlation_matrix.values[edges[:, 0], edges[:, 1]]

        edge_index = torch.tensor(edges.T, dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float)
        gene_indices = {gene: idx for idx, gene in enumerate(gene_expression_data.index)}
        
        G = nx.Graph()
        G.add_nodes_from(range(len(gene_indices)))

        # Add edges to the graph
        for i, j, w in zip(edges[:, 0], edges[:, 1], weights):
            if i != j:  # Avoid self-loops
                G.add_edge(i, j, weight=w)
        
        # Step 2: Sample subgraphs
        for tf_gene in self.tf_genes:
            if tf_gene in gene_indices:
                tf_idx = gene_indices[tf_gene]
                
                subgraph_nodes = set(nx.bfs_tree(G, tf_idx, depth_limit=None).nodes)
                if len(subgraph_nodes) > 100:
                    subgraph_nodes = set(list(subgraph_nodes)[:100])

                subgraph = G.subgraph(subgraph_nodes)
                subgraph_data = from_networkx(subgraph)
                subgraph_data.x = torch.tensor(gene_expression_data.loc[gene_expression_data.index[list(subgraph.nodes)]].values, dtype=torch.float)
                subgraph_data.edge_weight = torch.tensor([G[u][v]['weight'] for u, v in subgraph.edges], dtype=torch.float)
                subgraph_data.num_nodes = len(subgraph.nodes)
                subgraph_data.gene_indices = torch.tensor(list(subgraph.nodes), dtype=torch.long)
                data_list.append(subgraph_data)

        # Step 3: Create regulation matrix
        regulation_data = pd.read_csv(self.raw_paths[1])
        regulation_data['Type'] = 1
        num_genes = len(gene_expression_data.index)
        regulation_matrix = np.zeros((num_genes, num_genes))

        for _, row in regulation_data.iterrows():
            gene1, gene2, regulation_type = row
            if gene1 in gene_indices and gene2 in gene_indices:
                idx1 = gene_indices[gene1]
                idx2 = gene_indices[gene2]
                regulation_matrix[idx1, idx2] = regulation_type
                #print(regulation_matrix.shape)
        G_regulation = nx.from_numpy_matrix(regulation_matrix, create_using=nx.DiGraph)

        # Step 4: Create unique label graphs for each subgraph
        label_graphs = []
        for subgraph_data in data_list:
            subgraph_indices = subgraph_data.gene_indices.tolist()
            subgraph_regulation_matrix = regulation_matrix[np.ix_(subgraph_indices, subgraph_indices)]
            subgraph_G = nx.from_numpy_matrix(subgraph_regulation_matrix, create_using=nx.DiGraph)
            label_graph = from_networkx(subgraph_G)
            label_graph.x = subgraph_data.x
            label_graph.edge_weight = torch.tensor([subgraph_G[u][v]['weight'] for u, v in subgraph_G.edges], dtype=torch.float)
            #label_graphs.append(label_graph)
            # Assign the label graph to the subgraph's y attribute
            subgraph_data.y = label_graph
            label_graphs.append(subgraph_data)
        print(len(label_graphs))
        # Step 4: Save the processed data
        data, slices = self.collate(label_graphs)

        # Step 5: Combine the data into PyTorch Geometric dataset
        #data, slices = self.collate(label_graphs)
        torch.save((data, slices), self.processed_paths[0])

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

class GeneExpressionDataset(Dataset):
    def __init__(self,root, gene_expression_file, tf_genes, regulation_file):
        self.gene_expression_data = pd.read_csv(gene_expression_file, index_col=0)
        self.tf_genes = tf_genes
        self.regulation_data = pd.read_csv(regulation_file)
        self.regulation_data['Type'] = 1
        # Create the weighted gene co-expression network
        self.correlation_matrix = self.gene_expression_data.corr()
        self.gene_indices = {gene: idx for idx, gene in enumerate(self.gene_expression_data.index)}

        # Create the NetworkX graph and add all nodes
        self.G = nx.Graph()
        self.G.add_nodes_from(range(len(self.gene_indices)))

        # Add edges based on the correlation matrix
        edges = np.array(np.nonzero(self.correlation_matrix)).T
        weights = self.correlation_matrix.values[edges[:, 0], edges[:, 1]]
        for i, j, w in zip(edges[:, 0], edges[:, 1], weights):
            if i != j:  # Avoid self-loops
                self.G.add_edge(i, j, weight=w)

        # Process all subgraphs and labels
        self.subgraphs = []
        self.labels = []
        self.process_subgraphs()

    def process_subgraphs(self):
        for tf_gene in self.tf_genes:
            if tf_gene in self.gene_indices:
                tf_idx = self.gene_indices[tf_gene]

                subgraph_nodes = set(nx.bfs_tree(self.G, tf_idx, depth_limit=None).nodes)
                if len(subgraph_nodes) > 100:
                    subgraph_nodes = set(list(subgraph_nodes)[:100])

                subgraph = self.G.subgraph(subgraph_nodes)
                subgraph_data = from_networkx(subgraph)
                subgraph_data.x = torch.tensor(self.gene_expression_data.loc[self.gene_expression_data.index[list(subgraph.nodes)]].values, dtype=torch.float)
                subgraph_data.edge_weight = torch.tensor([self.G[u][v]['weight'] for u, v in subgraph.edges], dtype=torch.float)
                subgraph_data.num_nodes = len(subgraph.nodes)
                subgraph_data.gene_indices = torch.tensor(list(subgraph.nodes), dtype=torch.long)

                # Create unique label graph for the subgraph
                num_genes = len(self.gene_expression_data.index)
                regulation_matrix = np.zeros((num_genes, num_genes))
                for _, row in self.regulation_data.iterrows():
                    gene1, gene2, regulation_type = row
                    if gene1 in self.gene_indices and gene2 in self.gene_indices:
                        idx1 = self.gene_indices[gene1]
                        idx2 = self.gene_indices[gene2]
                        regulation_matrix[idx1, idx2] = regulation_type

                subgraph_indices = subgraph_data.gene_indices.tolist()
                subgraph_regulation_matrix = regulation_matrix[np.ix_(subgraph_indices, subgraph_indices)]
                subgraph_G = nx.from_numpy_matrix(subgraph_regulation_matrix, create_using=nx.DiGraph)
                label_graph = from_networkx(subgraph_G)
                label_graph.edge_weight = torch.tensor([subgraph_G[u][v]['weight'] for u, v in subgraph_G.edges], dtype=torch.float)
                print(label_graph)
                subgraph_data.y = label_graph
                self.subgraphs.append(subgraph_data)

    def __len__(self):
        return len(self.subgraphs)

    def __getitem__(self, idx):
        return self.subgraphs[idx]
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataListLoader
from torch_geometric.sampler import NumNeighbors
from torch_geometric.utils import dense_to_sparse, k_hop_subgraph,to_dense_adj,degree

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
        return ['DatasetCombined.pt']

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
        #print(regulation_matrix.loc["CEBPB",tf_gene],gene_indices["CEBPB"])
        print(regulation_matrix.shape)
        # Step 2: Sample subgraphs
        for tf_gene in self.tf_genes:
            if tf_gene in gene_indices:
                tf_idx = gene_indices[tf_gene]
                print(tf_idx)
                print(correlation_matrix[tf_idx])
                
                # Get 1-hop and 2-hop neighbors
                num_hops = 1
                while num_hops<100:
                    subgraph_node_idx, subgraph_edge_index, _, _ = k_hop_subgraph(tf_idx, num_hops=num_hops, edge_index=full_graph.edge_index, num_nodes=num_genes)
                    if len(subgraph_node_idx) >= 100:
                        break
                    print(num_hops,len(subgraph_node_idx))
                    num_hops += 1
                
                #if len(subgraph_node_idx) < 500:
                #    continue
                #subgraph_node_idx, subgraph_edge_index, _, _ = k_hop_subgraph(tf_idx, num_hops=2, edge_index=full_graph.edge_index, num_nodes=num_genes)
                print(subgraph_edge_index)
                first_hop_neighbors = full_graph.edge_index[1][full_graph.edge_index[0] == tf_idx]
                first_hop_neighbors = first_hop_neighbors.unique()
                print(len(first_hop_neighbors))
                first_hop_neighbors = first_hop_neighbors[first_hop_neighbors!= tf_idx]
                print(len(first_hop_neighbors))
                second_hop_neighbors = subgraph_node_idx[~torch.isin(subgraph_node_idx,first_hop_neighbors) & (subgraph_node_idx != tf_idx)]
                
                # Combine TF, first hop, and second hop neighbors to get exactly 100 nodes
                if len(first_hop_neighbors) + 1 >= 100:
                    subgraph_node_idx = torch.cat([torch.tensor([tf_idx]), first_hop_neighbors[:99]])
                else:
                    remaining_nodes = 100 - (len(first_hop_neighbors) + 1)
                    subgraph_node_idx = torch.cat([torch.tensor([tf_idx]), first_hop_neighbors, second_hop_neighbors[:remaining_nodes]])
                
                subgraph_node_set = set(subgraph_node_idx.tolist())
                print(torch.isin(tf_idx,subgraph_node_idx))
                # Remap the subgraph node indices
                subgraph_node_idx = subgraph_node_idx.unique()
                print(len(subgraph_node_idx))
                node_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph_node_idx.tolist())}

                # Filter edges to keep only those within the subgraph nodes
                mask = torch.isin(subgraph_edge_index[0], subgraph_node_idx) & torch.isin(subgraph_edge_index[1], subgraph_node_idx)
                subgraph_edge_index = subgraph_edge_index[:, mask]
                subgraph_edge_index = torch.tensor([[node_idx_map[i.item()], node_idx_map[j.item()]] for i, j in zip(subgraph_edge_index[0], subgraph_edge_index[1])], dtype=torch.long).t().contiguous()
                # Get the corresponding edge weights for the filtered edges
                subgraph_edge_weight = full_graph.edge_weight[
                    torch.isin(full_graph.edge_index[0], subgraph_node_idx) & 
                    torch.isin(full_graph.edge_index[1], subgraph_node_idx)
                ]
                print(subgraph_edge_index)
                exp_x = full_graph.x[subgraph_node_idx]
                #minm_x = self.min_max(exp_x)
                zscore = self.z_score_per_cell(exp_x)
                print("zscore shape",zscore.shape)
                subgraph_x = torch.column_stack((zscore,Is_tf[subgraph_node_idx]))
                
                #subgraph_x = full_graph.x[subgraph_node_idx]
                
                subgraph_data = Data(x=subgraph_x, edge_index=subgraph_edge_index, edge_weight=subgraph_edge_weight)
                #print("subgraph:",subgraph_data)
                # Step 3: Create unique label graph for the subgraph
                common_neigh = self.common_neighbors(subgraph_data.edge_index, len(subgraph_node_idx))

                jaccard = self.jaccard_coefficient(subgraph_data.edge_index, len(subgraph_node_idx))

                adamic_adar = self.adamic_adar_index(subgraph_data.edge_index, len(subgraph_node_idx))
                preferential_attach = self.preferential_attachment(subgraph_data.edge_index, len(subgraph_node_idx))
                print(subgraph_data.edge_weight.shape,common_neigh[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1).shape)
                edge_weight1 = torch.cat([subgraph_data.edge_weight.unsqueeze(1),
                            common_neigh[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1).float(),
                            jaccard[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1),
                            adamic_adar[subgraph_data.edge_index[0], subgraph_data.edge_index[1]].unsqueeze(1),
                            preferential_attach[subgraph_data.edge_index[0],subgraph_data.edge_index[1]].unsqueeze(1)], dim=1)

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
                if len(subgraph_regulation_edge_weight)==0:
                    continue
                label_graph = Data(edge_index=subgraph_regulation_edge_index, edge_weight=subgraph_regulation_edge_weight)
                #print(label_graph)
                # Assign the label graph to the subgraph's y attribute
                subgraph_data.y = label_graph.edge_index

                #subgraph_Is_TF =Is_tf[subgraph_node_idx]
                #Z_exp_acrossgenes = self.z_score(subgraph_x.t())
                
                #Z_exp_acrosscell = self.z_score_dim1(subgraph_x)
                #minmax_exp_acrosscell = self.min_max_dim0(Z_exp_acrosscell)
                #print(minm_acrossgenes.shape,minmax_exp_acrosscell.shape)
                #exp_mat_df = torch.as_tensor(Exp_data_df.to_numpy())
                

                #node_attr =  torch.column_stack((subgraph_Is_TF,minmax_exp_acrosscell,))
                self.data_list.append(subgraph_data)

        # Step 4: Save the processed data
        #data, slices = self.collate(data_list)
        torch.save(self.data_list, self.processed_paths[0])

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        
        data = self.data_list[idx].x
          
          
        edges = self.data_list[idx].edge_index
        #edge_weights = self.data_list[idx].edge_weight
        edge_weights = self.data_list[idx].edge_weight[:,0].unsqueeze(1)
        pos_edges = self.data_list[idx].y
        
        #gsum = self.SubaGraphs[idx].gsum
        #item = next(GCENdataset)
        return [data,edges,edge_weights, pos_edges]
# Usage
'''
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