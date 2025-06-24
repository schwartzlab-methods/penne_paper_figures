import os
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count

def preprocess_stimage_1k4m(base_dir: str):
    """
    Pre-process the STimage-1k4m dataset to only include human samples.
    """
    annotation = pd.read_csv(os.path.join(base_dir, "meta", "meta_all_gene02122025.csv"))
    human_samples = annotation[annotation["species"] == "human"]["slide"].tolist()
    # remove non-human images
    for img in os.listdir(os.path.join(base_dir, "image")):
        name = img.split(".")[0]
        if name not in human_samples:
            os.remove(os.path.join(base_dir, "image", img))
    # remove non-human gene expression files
    for exp in os.listdir(os.path.join(base_dir, "gene_exp")):
        name = exp.split("_count.csv")[0]
        if name not in human_samples:
            os.remove(os.path.join(base_dir, "gene_exp", exp))
    # remove non-human coordinate files
    for coor in os.listdir(os.path.join(base_dir, "coord")):
        name = coor.split("_coord.csv")[0]
        if name not in human_samples:
            os.remove(os.path.join(base_dir, "coord", coor))
    print("Pre-processing complete. Only human samples are retained.")

def find_common_genes(csv_files: str):
    """
    Find common genes in the given csv files.
    """
    common_genes = None
    for file in csv_files:
        df = pd.read_csv(file)
        genes = set(df.columns)
        if common_genes is None:
            common_genes = genes
        else:
            common_genes.intersection_update(genes)
    return sorted(list(common_genes))

def process_sample(args):
    """
    Process a single sample to create patches and save them.
    """
    sample, base_dir, common_genes = args
    try:
        image = Image.open(os.path.join(base_dir, "image", f"{sample}.png"))
        exp_df = pd.read_csv(os.path.join(base_dir, "gene_exp", f"{sample}_count.csv"))
        exp_df = exp_df[common_genes]
        coor_df = pd.read_csv(os.path.join(base_dir, "coord", f"{sample}_coord.csv"))
        patch_names = exp_df["Unamed: 0"].tolist()
        coor_df = coor_df.drop(columns=["Unamed: 0"])
        exp_df = exp_df.drop(columns=["Unamed: 0"])
        for i, patch_name in enumerate(patch_names):
            patch = exp_df.iloc[i].values
            x_centre, y_centre, radius = coor_df.iloc[i][["xaxis", "yaxis", "r"]]
            x = int(x_centre - radius)
            y = int(y_centre - radius)
            if x < 0 or y < 0 or x + 2 * radius > image.width or y + 2 * radius > image.height:
                print(f"Skipping patch {patch_name} for sample {sample} due to out of bounds coordinates.")
                continue
            patch_image = image.crop((x, y, x + 2 * radius, y + 2 * radius))
            patch_image.save(os.path.join(base_dir, "patch_images", f"{sample}_{patch_name}.png"))
            np.save(os.path.join(base_dir, "patches_exp", f"{sample}_{patch_name}.npy"), patch)
    except Exception as e:
        print(f"Error processing sample {sample}: {e}")

def create_patch_matrix(base_dir: str, common_genes: list):
    """
    Create a matrix with only the common genes using multiprocessing.
    """
    sample_names = [f.split(".")[0] for f in os.listdir(os.path.join(base_dir, "image")) if f.endswith('.png')]
    args = [(sample, base_dir, common_genes) for sample in sample_names]
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(process_sample, args), total=len(sample_names)))
    print("Patch matrix creation complete.")

def main():
    parser = argparse.ArgumentParser(description="Generate patch expression numpy file for STimage-1k4m dataset")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory of the STimage-1k4m dataset")
    args = parser.parse_args()

    base_dir = args.base_dir
    os.makedirs(os.path.join(base_dir, "patch_images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "patches_exp"), exist_ok=True)
    preprocess_stimage_1k4m(base_dir)

    exp_files = [os.path.join(base_dir, "gene_exp", f) for f in os.listdir(os.path.join(base_dir, "gene_exp")) if f.endswith('.csv')]
    common_genes = find_common_genes(exp_files)
    create_patch_matrix(base_dir, common_genes)

if __name__ == "__main__":
    main()
