import pandas as pd
import sys
# List of CSV file names
#csv_files = ['../Data/sc-RNA-seq/hESC/hESC-ChIP-seq-network.csv', '../Data/sc-RNA-seq/hESC/Non-specific-ChIP-seq-network.csv', '../Data/sc-RNA-seq/hESC/STRING-network.csv']
csv_files = [sys.argv[1], sys.argv[2], sys.argv[3]]
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
combined_df.to_csv(sys.argv[4], index=False)

