import os
import subprocess
import pandas as pd

df = pd.read_csv("/data/GRNformer/Data/hESC.csv")
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
        '''
        if arg5 == 1:
            result = subprocess.run(
            [   
                "/root/miniforge3/envs/grnformer/bin/python",
                "/data/GRNformer/GenerateInput.py",
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
                "/data/GRNformer/GenerateInput.py",
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
        
        '''
            

        #    /root/miniforge3/envs/grnformer/bin/python /data/GRNformer/GenerateInput.py --expFile "$arg1" --netFile "$arg2" --geneOrderingFile "$arg3" \
        #--TFFile "$arg4" --TFs --numGenes "$arg6" --outPrefix "$arg7"
        #else
        #    /root/miniforge3/envs/grnformer/bin/python /data/GRNformer/GenerateInput.py --expFile "$arg1" --netFile "$arg2" --geneOrderingFile "$arg3" \
        #--TFFile "$arg4" --numGenes "$arg6" --outPrefix "$arg7"
        #fi
    # result = subprocess.run(
        #     [
        #         "/root/miniforge3/envs/grnformer/bin/python",
        #         "/data/GRNformer/Inference.py",
        #         "--save_dir",
        #         "Evaluation_"+arg8+"_equalneg_0.5",
        #         "--root",
        #         "Data/sc-RNA-seq/mHSC-L",
        #         "--expFile",
        #         arg7+"-ExpressionData.csv",
        #         "--netFile",
        #         arg7+"-network1.csv",
        #         "--TFspecies",
        #         "mouse",
        #         "--outPrefix",
        #         arg7
                
        #     ],
        #     check=True,
        #     # timeout=900,
        #     # capture_output=True,
        #     text=True
        # )
        result = subprocess.run(
            [
                "/root/miniforge3/envs/grnformer/bin/python",
                "/data/GRNformer/Beelinegrnfomerinference.py",
                "--expFile",
                arg7+"-ExpressionData.csv",
                "--netFile",
                arg7+"-network1.csv",
                "--outPrefix",
                arg8
                
            ],
            check=True,
            # timeout=900,
            # capture_output=True,
            text=True
        )
    except Exception as e:
        print("Failed to process:", arg7)
        print(e)
