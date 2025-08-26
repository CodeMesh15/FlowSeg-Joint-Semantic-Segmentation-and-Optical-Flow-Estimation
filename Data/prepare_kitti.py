# /data/prepare_kitti.py

import os
import argparse
import zipfile
from tqdm import tqdm

# This is a conceptual script to guide you.
# You will need to add robust logic for downloading and file matching.

def download_file(url, target_dir):
    """Placeholder for a function to download a file with a progress bar."""
    print(f"INFO: Downloading from {url} to {target_dir}...")
    # Example: Use requests library to download
    # with requests.get(url, stream=True) as r:
    #     r.raise_for_status()
    #     with open(os.path.join(target_dir, 'file.zip'), 'wb') as f:
    #         for chunk in r.iter_content(chunk_size=8192):
    #             f.write(chunk)
    print("INFO: Download complete (placeholder).")

def unzip_file(zip_path, extract_to):
    """Unzips a file to a specified directory."""
    print(f"INFO: Unzipping {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("INFO: Unzipping complete.")

def main(args):
    """Main function to orchestrate data preparation."""
    
    # Create directories
    os.makedirs(args.download_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Step 1: Download the data (manual or scripted) ---
    print("--- Starting Data Download ---")
    print("Please download the following files manually or implement download logic:")
    print("1. Raw Data (e.g., 2011_09_26_drive_0005_sync.zip) from the KITTI Raw Data page.")
    print("2. Optical Flow Data (data_scene_flow.zip) from the KITTI 2015 Benchmark.")
    print("3. Semantic Segmentation Data (semantic-kitti.zip) from the SemanticKITTI page.")
    # download_file('URL_TO_FLOW_DATA', args.download_dir)
    # download_file('URL_TO_SEMANTIC_DATA', args.download_dir)
    
    # --- Step 2: Unzip the files ---
    print("\n--- Starting Unzipping ---")
    # Example:
    # unzip_file(os.path.join(args.download_dir, 'data_scene_flow.zip'), args.output_dir)
    
    # --- Step 3: Align and Organize the data ---
    print("\n--- Starting Data Organization (Manual step) ---")
    print("This is the most critical and project-specific step.")
    print("You must write a script to align the data. The goal is to create a final, clean dataset directory with a structure like this:")
    print("""
    /final_dataset/
    |-- /images/
    |   |-- 000000_10.png
    |   |-- 000000_11.png
    |   |-- ...
    |-- /flow/
    |   |-- 000000_10.png
    |   |-- ...
    |-- /segmentation/
    |   |-- 000000_10.png
    |   |-- ...
    """)
    print("This requires matching filenames and frame numbers across the different unzipped directories.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare KITTI data for FlowSeg.")
    parser.add_argument('--download_dir', type=str, required=True, help='Directory to download zip files to.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to extract and organize the final dataset.')
    
    args = parser.parse_args()
    main(args)
