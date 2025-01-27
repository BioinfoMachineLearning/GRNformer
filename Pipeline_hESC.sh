#!/bin/bash

# Read the header line to get column names
header=$(head -n 1 /home/aghktb/GRNformer/Data/hESC.csv)

# Loop through each line in the CSV (skipping the header)
tail -n +2 /home/aghktb/GRNformer/Data/hESC.csv | while IFS=, read -r arg1 arg2 arg3 arg4 arg5 arg6 arg7
do
    arg8=$(basename "$arg7")
   #  mkdir -p "inputs/examples/$arg8"
    ## Run the /root/miniforge3/envs/grnformer/bin/python script with arguments from the CSV row
   #  if [ "$arg5" -eq 1 ]; then

   #     ~/miniconda3/envs/grnformer/bin/python /home/aghktb/GRNformer/GenerateInput.py --expFile "$arg1" --netFile "$arg2" --geneOrderingFile "$arg3" \
   #  --TFFile "$arg4" --TFs --numGenes "$arg6" --outPrefix "inputs/examples/$arg8/"
   #  else
   #     ~/miniconda3/envs/grnformer/bin/python /home/aghktb/GRNformer/GenerateInput.py --expFile "$arg1" --netFile "$arg2" --geneOrderingFile "$arg3" \
   #  --TFFile "$arg4" --numGenes "$arg6" --outPrefix "inputs/examples/$arg8/"
   #  fi
   #  cp "$arg4" "inputs/examples/$arg8/TFs.csv"
    #cp "$arg3" "inputs/examples/$arg8/PseudoTime.csv"
    # basename "$arg7"
    ~/miniconda3/envs/grnformer/bin/python /home/aghktb/GRNformer/Beelinegrnfomerinference.py  --expFile "inputs/examples/$arg8/ExpressionData.csv" --netFile "inputs/examples/$arg8/refNetwork.csv" \
    --outPrefix "outputs/examples/$arg8" 
done