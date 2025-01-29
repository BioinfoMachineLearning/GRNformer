import pandas as pd
import sys
import argparse
# List of CSV file names
#csv_files = ['../Data/sc-RNA-seq/hESC/hESC-ChIP-seq-network.csv', '../Data/sc-RNA-seq/hESC/Non-specific-ChIP-seq-network.csv', '../Data/sc-RNA-seq/hESC/STRING-network.csv']

# Set up argument parser
parser = argparse.ArgumentParser(description='Combine multiple CSV files into one.')
parser.add_argument('--cell-type-network', help='cell type netwok csv to combine')
parser.add_argument('--non-specific-network', help='non specific network csv file name')
parser.add_argument('--string-network', help='string network csv file name')
parser.add_argument('--output-file', help='output file name')
args = parser.parse_args()

csv_files = [args.cell_type_network, args.non_specific_network, args.string_network]
# Create an empty dataframe to store the combined data
combined_df = pd.DataFrame()

# Read each CSV file and append to the combined dataframe
for fi in csv_files:
    print(fi)
    df = pd.read_csv(fi,usecols=["Gene1", "Gene2"])
    print(df)
    combined_df = pd.concat([combined_df, df], ignore_index=True)
print(combined_df.shape)
# Remove duplicate rows
combined_df.drop_duplicates(inplace=True)
print(combined_df.shape)
# Write the combined dataframe to a new CSV file
combined_df.to_csv(args.output_file, index=False)

