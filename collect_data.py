import shutil
import requests
import zipfile
import os
import argparse
import wget

# Parse the command line arguments
parser = argparse.ArgumentParser(description='Download and extract data.')
parser.add_argument('--data_dir', type=str, required=True, help='Directory to extract the data to')
#parser.add_argument('--url', type=str, default="https://zenodo.org/records/3701939/files/BEELINE-data.zip?download=1", help='URL to download the data from')
parser.add_argument('--extract_folder', type=str, default="sc-RNAseq", help='Folder to extract from the zip file')

args = parser.parse_args()
# Define the URL and the local filename
url = "https://zenodo.org/records/3701939/files/BEELINE-data.zip?download=1"
local_zip_file = "BEELINE-data.zip"
extract_folder = "BEELINE-data/inputs/scRNA-Seq"

# Download the file using wget
#wget.download(args.url, local_zip_file)

# Extract the zip file
temp_extract_path = args.data_dir + "_temp"  # Temporary extraction directory

with zipfile.ZipFile(local_zip_file, "r") as zip_ref:
    zip_ref.extractall(temp_extract_path)  # Extract everything temporarily

    # Path of the folder inside the extracted contents
    source_folder = os.path.join(temp_extract_path, extract_folder)
    final_folder = os.path.join(args.data_dir)

    if os.path.exists(source_folder):
        # Move only the specific folder to the final location
        shutil.move(source_folder, final_folder)
        print(f"Extracted '{extract_folder}' to '{final_folder}'")
    else:
        print(f"Folder '{extract_folder}' not found in ZIP!")

    # Clean up the temporary extraction
    shutil.rmtree(temp_extract_path, ignore_errors=True)

os.remove(local_zip_file)
# Clean up the zip file
ground_truth_url = "https://zenodo.org/records/3701939/files/BEELINE-Networks.zip?download=1"
local_zip_file = "BEELINE-Networks.zip"
extract_folder = args.data_dir+"-Networks"
#wget.download(ground_truth_url, local_zip_file)
with zipfile.ZipFile(local_zip_file, "r") as zip_ref:
    zip_ref.extractall(extract_folder)
    print(f"Extracted ground truth networks to '{args.data_dir}'")


os.remove(local_zip_file)

print(f"Downloaded and extracted {extract_folder} folder successfully.")

