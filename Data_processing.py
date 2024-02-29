from pandas import read_csv


'''
Once we obtained the pseduotime values for the cells in each dataset, 
we computed which genes had varying expression values across pseudotime. 
We used the general additive model implemented in the ‘gam’ R package to compute the variance and the P value of this variance. 
We used the Bonferroni method to correct for testing multiple hypotheses. 
 We selected genes for GRN inference in two different ways.

i.We considered all genes with a P value less than 0.01. We selected variance thresholds so that we obtained 500 and 1,000 highly varying genes. We recorded the number of TFs in these sets.

ii.We started by including all TFs whose variance had P value at most 0.01. 
Then, we added 500 and 1,000 additional genes as in the previous option. This approach enabled the GRN methods to consider TFs that may have a modest variation in gene expression but still regulate their targets.

After applying a GRN inference algorithm to a dataset, we only considered interactions outgoing from a TF in further evaluation.
Source: Pratapa, A., Jalihal, A.P., Law, J.N. et al. Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data. Nat Methods 17, 147–154 (2020). 
https://doi.org/10.1038/s41592-019-0690-6



DATASET_DIR = "/home/aghktb/GRN/GRNformer"
train = DATASET_DIR+'/Data/sc-RNA-seq/hHep/GeneOrdering.csv'
TFs = DATASET_DIR+'/Data/TFHumans.csv'
GeneOrder = read_csv(train)
HumTfs = read_csv(TFs,header=None)
print(HumTfs.head())
GeneOrder1 = GeneOrder.loc[GeneOrder["VGAMpValue"]<0.01,:]
print(len(GeneOrder1))

TFGenes = GeneOrder[GeneOrder['Unnamed: 0'].isin(HumTfs[0]) &  (GeneOrder['VGAMpValue'] <= 0.01)]
print(len(TFGenes))
#GeneOrder2 = GeneOrder.loc[GeneOrder["VGAMpValue"]<=0.01,:]
#print(len(GeneOrder2))


###################
Regulation_network  = read_csv(DATASET_DIR+'/Data/sc-RNA-seq/hHep/HepG2-ChIP-seq-network.csv')

regNet_of_GeneOrdr1 = Regulation_network[Regulation_network['Gene1'].isin(GeneOrder1['Unnamed: 0']) & Regulation_network['Gene2'].isin(GeneOrder1['Unnamed: 0'])]
print(len(regNet_of_GeneOrdr1))

regNet_of_GeneOrdr2 = Regulation_network[Regulation_network['Gene1'].isin(TFGenes['Unnamed: 0']) & Regulation_network['Gene2'].isin(TFGenes['Unnamed: 0'])]
print(len(regNet_of_GeneOrdr2))

##################
'''
DATASET_DIR = "/home/aghktb/GRN/GRNformer"
TFs = read_csv(DATASET_DIR+'/Data/TFHumans.csv',header=None)
Genes = read_csv(DATASET_DIR+'/Data/Human_tfs.csv',index_col=0)
print(Genes['Symbol'].head())

tfingenes = TFs.isin(Genes['Symbol']).any()
print(tfingenes)