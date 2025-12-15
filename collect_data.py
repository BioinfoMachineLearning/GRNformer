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

# Normalize data directory (remove trailing slashes)
data_dir = args.data_dir.rstrip("/")

# Define the URL and the local filename
url = "https://zenodo.org/records/3701939/files/BEELINE-data.zip?download=1"
local_zip_file = "BEELINE-data.zip"
extract_folder = "BEELINE-data/inputs/scRNA-Seq"

# Download the file using wget
wget.download(url, local_zip_file)

# Extract the zip file
temp_extract_path = data_dir + "_temp"  # Temporary extraction directory

with zipfile.ZipFile(local_zip_file, "r") as zip_ref:
    zip_ref.extractall(temp_extract_path)  # Extract everything temporarily

    # Path of the folder inside the extracted contents
    source_folder = os.path.join(temp_extract_path, extract_folder)
    final_folder = data_dir  # e.g. ./Data/scRNA-seq

    if os.path.exists(source_folder):
        # Ensure the final data directory exists
        os.makedirs(final_folder, exist_ok=True)

        # Move each dataset folder/file from scRNA-Seq into Data/scRNA-seq
        # This avoids creating an extra "scRNA-Seq" level and makes reruns safe.
        for item in os.listdir(source_folder):
            src_path = os.path.join(source_folder, item)
            dst_path = os.path.join(final_folder, item)

            # If something already exists at the destination, remove it first
            if os.path.isdir(dst_path):
                shutil.rmtree(dst_path)
            elif os.path.isfile(dst_path):
                os.remove(dst_path)

            shutil.move(src_path, dst_path)

        print(f"Extracted '{extract_folder}' contents to '{final_folder}'")
    else:
        print(f"Folder '{extract_folder}' not found in ZIP!")

    # Clean up the temporary extraction
    shutil.rmtree(temp_extract_path, ignore_errors=True)

os.remove(local_zip_file)

# Download and extract ground truth network ZIP
ground_truth_url = "https://zenodo.org/records/3701939/files/BEELINE-Networks.zip?download=1"
local_zip_file = "BEELINE-Networks.zip"
network_dir = data_dir + "-Networks"  # e.g. ./Data/scRNA-seq-Networks

# Clean existing networks directory if present
if os.path.exists(network_dir):
    shutil.rmtree(network_dir)

os.makedirs(network_dir, exist_ok=True)

wget.download(ground_truth_url, local_zip_file)

with zipfile.ZipFile(local_zip_file, "r") as zip_ref:
    zip_ref.extractall(network_dir)
    print(f"Extracted ground truth networks to '{network_dir}'")

os.remove(local_zip_file)

print(f"Downloaded and extracted networks to '{network_dir}' successfully.")

