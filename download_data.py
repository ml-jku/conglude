import argparse
import os
import requests
import zipfile

# Create argument parser for optional dataset selection
parser = argparse.ArgumentParser(description="Download evaluation datasets from Zenodo and extract them.")

# Optional argument: name of a specific dataset to download
parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset to download (default: download all datasets).")

# Parse arguments
args = parser.parse_args()

# Zenodo record containing all evaluation datasets
record_id = 18933183
url = f"https://zenodo.org/api/records/{record_id}"

# Request metadata for the Zenodo record
r = requests.get(url)
data = r.json()

# Iterate over all files
for file in data["files"]:
    download_url = file["links"]["self"]
    filename = file["key"]

    # Download file if no dataset was specified, or the filename matches the requested dataset
    if args.dataset_name is None or filename[:-4] == args.dataset_name:

        # Download the dataset archive
        print(f"Downloading dataset {filename[:-4]}.")
        file_data = requests.get(download_url)

        with open(f"data/datasets/test_datasets/{filename}", "wb") as f:
            f.write(file_data.content)

        # Extract dataset contents
        print(f"Unzipping dataset {filename[:-4]}.")
        zip_path = f"data/datasets/test_datasets/{filename}"
        extract_folder = f"data/datasets/test_datasets/{filename[:-4]}"

        os.makedirs(extract_folder, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_folder)

        # Delete the .zip file
        os.remove(zip_path)