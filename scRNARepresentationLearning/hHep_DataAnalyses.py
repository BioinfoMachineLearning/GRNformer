import umap 
import Dataset_copy as dt
import pandas as pd
import numpy as np
import umap.plot
import seaborn as sns
import glob
import os
import matplotlib.pyplot as plt
from pandas import read_csv,DataFrame,crosstab,unique
from scipy.stats import spearmanr

DATASET_DIR = "/home/aghktb/GRN/GRNformer"
Exp_data=DATASET_DIR+'/Data/sc-RNA-seq/hHep/ExpressionData.csv'
Exp_data_df = read_csv(Exp_data,index_col=0)
genes = Exp_data_df.index
#Apply asin transform on expression data
Exp_data_df = Exp_data_df.apply(np.arcsinh)
#Generate paiwaise pearson coorelation of genes

Exp_coe_data_np =DataFrame(np.corrcoef(Exp_data_df),index=genes,columns=genes)




Exp_label_pth = os.path.dirname(Exp_data)+"/HepG2-ChIP-seq-network.csv"
Exp_label_df = read_csv(Exp_label_pth)
Exp_label_df['Type'] = 1
genes_tar = unique(Exp_label_df[['Gene1', 'Gene2']].values.ravel('K'))
Exp_tar_adj = DataFrame(index=genes, columns=genes)

# Fill in the values based on df2
for index, row in Exp_label_df.iterrows():
        if(row['Gene1'] in genes)and (row['Gene2'] in genes):
            Exp_tar_adj.at[row['Gene1'], row['Gene2']] = row['Type']
Exp_tar_adj = Exp_tar_adj.fillna(0)




#n = len(genes)
flattened_matrix1 = Exp_coe_data_np.to_numpy().flatten()
flattened_matrix2 = Exp_tar_adj.to_numpy().flatten()

# Combine the flattened arrays into a 2D array (each as a row)
#combined_matrix = np.array([flattened_matrix1, flattened_matrix2])

# Compute the correlation coefficient matrix
from scipy.stats import pearsonr
from scipy.stats import pointbiserialr

correlation, p_value = pointbiserialr(flattened_matrix2, flattened_matrix1)
correlation_matrix = np.corrcoef(flattened_matrix1,flattened_matrix2)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True)
plt.title('Correlation Coefficient Matrix')
plt.savefig("coorelationplot.png")


# Calculate the overlap (element-wise multiplication of the two matrices)
co_expression_df=Exp_coe_data_np
regulation_df = Exp_tar_adj
overlap_matrix =  co_expression_df* regulation_df

# Summarize the overlap
print("Co-expression Matrix:\n", co_expression_df)
print("\nRegulation Matrix:\n", regulation_df)
print("\nOverlap Matrix:\n", overlap_matrix)

# Example analysis: count the number of regulatory interactions with high co-expression
threshold = 0.5
high_co_expression = (co_expression_df.abs() >= threshold)
high_co_expression_and_regulation = high_co_expression & (regulation_df == 1)

num_high_co_expression_and_regulation = high_co_expression_and_regulation.sum().sum()
print(f"\nNumber of regulatory interactions with high co-expression (|r| >= {threshold}):",
      num_high_co_expression_and_regulation)

# Plot the correlation matrix as a heatmap
##plt.figure(figsize=(8, 6))
#lt.scatter(correlation,p_value, cmap='coolwarm')
#plt.title('Correlation Coefficient Matrix')
#plt.savefig("coorelationplot.png")


from scipy.stats import pointbiserialr
results = []
# Function to check if a DataFrame column is constant
def is_constant(series):
    return series.nunique() == 1

# Check and compute correlation

for idx, co_expression_row in co_expression_df.iterrows():
      regulation_row = regulation_df.loc[idx, :]
      if is_constant(co_expression_row) or is_constant(regulation_row):
            continue
      else:
            
            correlation, p_value = pointbiserialr(co_expression_row, regulation_row)
            results.append((idx, correlation, p_value))

# Convert results to a DataFrame if needed
results_df = pd.DataFrame(results, columns=['Index', 'Correlation', 'P_value'])
print(results_df)
results_df.to_csv("correlation_df.csv")


