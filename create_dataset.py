import os
import pandas as pd
import torch
from src.datamodules.tf_walker_dataset_train import GeneExpressionDataset


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='Data/sc-RNA-seq', type=str)
    parser.add_argument('--dataset_list', default='train_list.csv', type=str)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    dataset_list = pd.read_csv('{dataset_dir}/{args.dataset_list}',header=None)[0].to_list()
    print(dataset_list)
    for i in dataset_list:
            root = os.path.abspath(f'{dataset_dir}/{i}')
            gene_expression_file = os.path.abspath(f'{dataset_dir}/{i}/ExpressionData.csv')
            tf_list = os.path.abspath(f'{dataset_dir}/{i}/TFs.csv')
            

            regulation_file = os.path.abspath(f'{dataset_dir}/{i}/{i}_combined.csv')
            dataset = GeneExpressionDataset(root,gene_expression_file,tf_list,regulation_file)

            print(len(dataset))
            train_size = int(0.70 * len(dataset))
            val_size = int(0.1 * len(dataset))
            test_size = len(dataset) - (train_size+val_size)
            dataset_train,dataset_valid,dataset_test = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

            train_indices = dataset_train.indices
            valid_indices = dataset_valid.indices
            test_indices = dataset_test.indices
            split_ind = os.path.abspath(f'{dataset_dir}/{i}/dataset_splits_combined.pt')
            torch.save({'train_indices': train_indices,'valid_indices':valid_indices, 'test_indices': test_indices}, split_ind)
