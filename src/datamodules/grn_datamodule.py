import glob
import os
import pandas as pd
import torch

from src.datamodules.tf_walker_dataset_train import GeneExpressionDataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader,ConcatDataset
from torch_geometric.loader import DataListLoader


class GRNDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_dir="",
        batch_size=8,
        num_workers=1,
        pin_memory=False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.root = [os.path.abspath('Data/sc-RNA-seq/hESC'),os.path.abspath('Data/sc-RNA-seq/hHep'),
            os.path.abspath('Data/sc-RNA-seq/mDC'),os.path.abspath('Data/sc-RNA-seq/mHSC-E'),
            os.path.abspath('Data/sc-RNA-seq/mHSC-GM')]
        self.gene_expression_file = [os.path.abspath('Data/sc-RNA-seq/hESC/ExpressionData.csv'),os.path.abspath('Data/sc-RNA-seq/hHep/ExpressionData.csv'),
                                os.path.abspath('Data/sc-RNA-seq/mDC/ExpressionData.csv'),os.path.abspath('Data/sc-RNA-seq/mHSC-E/ExpressionData.csv'),
                                os.path.abspath('Data/sc-RNA-seq/mHSC-GM/ExpressionData.csv')]
        tfhum_genes = pd.read_csv(os.path.abspath("Data/sc-RNA-seq/hESC/TFHumans.csv"),header=None)[0].to_list()
        tfmou_genes = pd.read_csv(os.path.abspath("Data/sc-RNA-seq/mDC/TFMouse.csv"))['TF'].to_list()
        self.TF_list = [tfhum_genes,tfhum_genes,tfmou_genes,tfmou_genes,tfmou_genes]
        # replace with actual TF gene names
        self.regulation_file = [os.path.abspath('Data/sc-RNA-seq/hESC/hESC_combined.csv'),os.path.abspath('Data/sc-RNA-seq/hHep/hHep_combined.csv'),
                        os.path.abspath('Data/sc-RNA-seq/mDC/mDC_combined.csv'), os.path.abspath('Data/sc-RNA-seq/mHSC-E/mHSC-E_combined.csv'),
                            os.path.abspath('Data/sc-RNA-seq/mHSC-GM/mHSC-GM_combined.csv')]
        self.split_ind = [os.path.abspath("Data/sc-RNA-seq/hESC/dataset_splits_combined.pt"),os.path.abspath("Data/sc-RNA-seq/hHep/dataset_splits_combined.pt"),
                    os.path.abspath("Data/sc-RNA-seq/mDC/dataset_splits_combined.pt") ,os.path.abspath("Data/sc-RNA-seq/mHSC-E/dataset_splits_combined.pt"),
                    os.path.abspath("Data/sc-RNA-seq/mHSC-GM/dataset_splits_combined.pt")]

    def setup(self, stage):
        All_train_dataset=[]
        All_valid_dataset=[]
        All_test_dataset=[]
        for i in range(0,len(self.root)):
            dataset = GeneExpressionDataset(self.root[i],self.gene_expression_file[i],self.TF_list[i],self.regulation_file[i])

            print(len(dataset))
            split_indices = torch.load(self.split_ind[i],weights_only=True)
            train_indices = split_indices['train_indices']
            valid_indices = split_indices['valid_indices']
            test_indices = split_indices['test_indices']

            # Create Subsets using the loaded indices
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            valid_dataset = torch.utils.data.Subset(dataset, valid_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)
            All_train_dataset.append(train_dataset)
            All_valid_dataset.append(valid_dataset)
            All_test_dataset.append(test_dataset)

        self.TrainDatasets = ConcatDataset(All_train_dataset)
        self.ValidDatasets = ConcatDataset(All_valid_dataset)
        self.TestDatasets = ConcatDataset(All_test_dataset)

    def train_dataloader(self):
        return DataListLoader(dataset=self.TrainDatasets, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataListLoader(dataset=self.ValidDatasets, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataListLoader(dataset=self.TestDatasets, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
