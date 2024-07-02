import umap 
import Dataset_copy as dt
import pandas as pd
import numpy as np
import umap.plot
import seaborn as sns
import glob
import os
import matplotlib.pyplot as plt
DATASET_DIR = "/home/aghktb/GRN/GCEN"
dataset = dt.GCENgraph(DATASET_DIR)

Gene_co_exp = [dataset[i][6].numpy() for i in range(0,len(dataset))]
train = glob.glob(DATASET_DIR+'/Data/Curated/**/**/Expression*.*', recursive=True)


#plot umap for gene co expression

map = umap.UMAP()
for i in range(0,len()):
    Exp_data = train[i]
    #Expt_file_name = os.path.basename(pest_data)
    #Read Expression data
    Exp_data_df = pd.read_csv(Exp_data,index_col=0)
    embedding = pd.DataFrame(map.fit_transform(Exp_data_df), columns = ['UMAP1','UMAP2'])
    
    sns_plot = sns.scatterplot(x='UMAP1', y='UMAP2', data=embedding,alpha=.1, linewidth=0, s=1)
    plt.scatter(embedding['UMAP1'],embedding['UMAP2'])

# Save PNG
sns_plot.figure.savefig(DATASET_DIR+'/scRNARepresentationLearning/umap_scatter_gcen_raw.png', bbox_inches='tight', dpi=500)




