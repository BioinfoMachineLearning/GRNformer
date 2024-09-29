import DatasetMaker.DatasetwithTFcenter as dt
from torch.utils.data import DataLoader
import pandas as pd
import torch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
BATCH_SIZE=1
DATASET_DIR = "C:/Users/aghktb/Documents/GRN/GRNformer"
root = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/hHep'
gene_expression_file = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/hHep/Filtered--ExpressionData.csv'
tf_genes = pd.read_csv("C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/hHep/TFHumans.csv",header=None)[0].to_list()
# replace with actual TF gene names
regulation_file = 'C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/hHep/Filtered--network1.csv'

#dataset = dt.GeneExpressionDataset(root,gene_expression_file,tf_genes,regulation_file)

#test_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_dataloader_workers)
#print(len(test_loader))

dataset_outputs = torch.load(DATASET_DIR+'/GRN_Predictions_hHepChipseq.pt')
#print(dataset_outputs)
genes = pd.read_csv(gene_expression_file,index_col=0).index
genes=genes.to_list()
threshold=0.9
grnpred = dataset_outputs[0][f'preds_grn']
print(grnpred.shape)
adj = torch.sigmoid(torch.matmul(grnpred, grnpred.t()))
print(adj)
grnprob = pd.DataFrame((adj>threshold).float().numpy())
grnprob.columns = genes
grnprob.index =genes
#print(grnprob.sum())


edge_index = (adj>threshold).float().nonzero().t().contiguous()
#print(edge_index.shape)
# Extract source and target indices from edge_index
source_indices = edge_index[0]
target_indices = edge_index[1]

# Map indices to gene names
source_genes = [genes[i] for i in source_indices]
target_genes = [genes[i] for i in target_indices]

# Create a DataFrame
df = pd.DataFrame({'Gene1': source_genes, 'Gene2': target_genes})
print(df)

df_reg = pd.read_csv(regulation_file)

#merge two DataFrames and create indicator column
df_all = df.merge(df_reg.drop_duplicates(), on=['Gene1','Gene2'],
                   how='left', indicator=True)

#create DataFrame with rows that exist in first DataFrame only
df1_only = df_all[df_all['_merge'] == 'left_only']

print(df1_only)
#df.to_csv("C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/hHep/PredictedGRNS.csv")
df1_only.to_csv("C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/hHep/PredictedGRNS_notingold.csv")

#tfs = pd.read_csv(tf_genes)
tf_mapping = pd.DataFrame({'Genes':genes})
tf_mapping['Map'] = ["tf" if x in tf_genes else "gene" for x in genes]
print(tf_mapping)
tf_mapping.to_csv("C:/Users/aghktb/Documents/GRN/GRNformer/Data/sc-RNA-seq/hHep/TFmapping.csv")