import os
import subprocess
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Run GRNformer evaluation pipeline")
parser.add_argument("--dataset_file", type=str, required=True, help="Path to the CSV file containing input parameters")
parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
args = parser.parse_args()
output_dir = args.output_dir
df = pd.read_csv(args.dataset_file)
for i, row in df.iterrows():
    
    arg1 = row["expression"]
    arg2 = row["network"]
    arg3 = row["geneorder"]
    arg4 = row["tffile"]
    arg5 = row["includetf"]
    arg6 = row["numgenes"]
    arg7 = row["outprefix"]
    arg8 = os.path.basename(arg7)
    
    try:
        ## Run the /root/miniforge3/envs/grnformer/bin/python script with arguments from the CSV row
        #if [ "$arg5" -eq 1 ]; then
        
        if arg5 == 1:
            result = subprocess.run(
            [   
                "/root/miniforge3/envs/grnformer/bin/python",
                "GenerateInput.py",
                "--expFile",
                arg1,
                "--netFile",
                arg2,
                "--geneOrderingFile",
                arg3,
                "--TFFile",
                arg4,
                "--TFs",
                "--numGenes",
                arg6,
                "--outPrefix",
                arg7
            ],
                check=True,
                # timeout=900,
                # capture_output=True,
                text=True
            )
        else:
            result = subprocess.run(
            [   
                "/root/miniforge3/envs/grnformer/bin/python",
                "GenerateInput.py",
                "--expFile",
                arg1,
                "--netFile",
                arg2,
                "--geneOrderingFile",
                arg3,
                "--TFFile",
                arg4,
                "--numGenes",
                arg6,
                "--outPrefix",
                arg7
            ],
                check=True,
                # timeout=900,
                # capture_output=True,
                text=True
            )
        arg8 = os.path.basename(arg7)

        result = subprocess.run(
            [
                "/root/miniforge3/envs/grnformer/bin/python",
                "eval_grn.py",
                "--exp_file",
                output_dir+"/"+arg8+"/ExpressionData.csv",
                "--tf_file",
                output_dir+"/"+arg8+"/TFs.csv",
                "--net_file",
                output_dir+"/"+arg8+"/refNetwork.csv",
                "--output_file",
                output_dir+"/"+arg8+"/predictedNetwork.csv",
                
            ],
            check=True,
            # timeout=900,
            # capture_output=True,
            text=True
        )
    except Exception as e:
        print("Failed to process:", arg7)
        print(e)
