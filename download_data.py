import argparse
import os
import shutil
import requests
import zipfile

# Create argument parser for optional dataset selection
parser = argparse.ArgumentParser(description="Download evaluation datasets from Zenodo and extract them.")

# Optional argument: name of a specific dataset to download
parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset to download (default: download all datasets).")

# Parse arguments
args = parser.parse_args()

DATASET_GROUPS = {
    "test": ["asd", "coach420", "dude", "holo4k", "kinobeads", "litpcba", "pdbbind_refined", "pdbbind_time", "posebusters"],
    "train": ["SB_train_val", "LB_train_val"],
    "train_val": ["SB_train_val", "LB_train_val"],
    "vs": ["dude", "litpcba"],
    "tf": ["kinobeads"],
    "pp": ["coach420", "holo4k", "pdbbind_refined"],
    "pr": ["asd", "pdbbind_time", "posebusters"],
}

TRAIN_DATASETS = {"SB_train_val", "LB_train_val"}

if args.dataset_name in DATASET_GROUPS:
    requested_datasets = set(DATASET_GROUPS[args.dataset_name])
elif args.dataset_name is not None:
    requested_datasets = {args.dataset_name}
else:
    requested_datasets = None

# Zenodo record containing all evaluation datasets
record_id = 20354834
url = f"https://zenodo.org/api/records/{record_id}"

# Request metadata for the Zenodo record
r = requests.get(url)
data = r.json()

# Iterate over all files
for file in data["files"]:
    download_url = file["links"]["self"]
    filename = file["key"]

    # Download file if no dataset was specified, or the filename matches the requested dataset(s)
    if requested_datasets is None or filename[:-4] in requested_datasets:

        subfolder = "train_val_datasets" if filename[:-4] in TRAIN_DATASETS else "test_datasets"
        extract_folder = f"data/datasets/{subfolder}/{filename[:-4]}"
        if os.path.isdir(extract_folder):
            print(f"Skipping {filename[:-4]} (already exists at {extract_folder}).")
            continue

        # Download the dataset archive
        print(f"Downloading dataset {filename[:-4]}.")
        file_data = requests.get(download_url)

        os.makedirs(f"data/datasets/{subfolder}", exist_ok=True)
        with open(f"data/datasets/{subfolder}/{filename}", "wb") as f:
            f.write(file_data.content)

        # Extract dataset contents
        print(f"Unzipping dataset {filename[:-4]}.")
        zip_path = f"data/datasets/{subfolder}/{filename}"

        os.makedirs(extract_folder, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_folder)

        # Delete the .zip file
        os.remove(zip_path)

        # Flatten nested directory (e.g. LB_train_val/LB_train_val/... -> LB_train_val/...)
        nested = os.path.join(extract_folder, filename[:-4])
        if os.path.isdir(nested):
            for item in os.listdir(nested):
                shutil.move(os.path.join(nested, item), extract_folder)
            os.rmdir(nested)