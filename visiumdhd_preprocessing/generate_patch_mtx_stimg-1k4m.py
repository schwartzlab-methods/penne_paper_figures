import os
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from multiprocessing import get_context

def preprocess_stimage_1k4m(base_dir: str):
    """
    Pre-process the STimage-1k4m dataset to only include human samples.
    """
    annotation = pd.read_csv(os.path.join(base_dir, "meta", "meta_all_gene02122025.csv"))
    human_samples = annotation[annotation["species"] == "human"]["slide"].tolist()
    # remove non-human images
    for img in os.listdir(os.path.join(base_dir, "Visium", "image")):
        name = img.split(".")[0]
        if name not in human_samples:
            os.remove(os.path.join(base_dir, "Visium", "image", img))
    # remove non-human gene expression files
    for exp in os.listdir(os.path.join(base_dir, "Visium", "gene_exp")):
        name = exp.split("_count.csv")[0]
        if name not in human_samples:
            os.remove(os.path.join(base_dir, "Visium", "gene_exp", exp))
    # remove non-human coordinate files
    for coor in os.listdir(os.path.join(base_dir, "Visium", "coord")):
        name = coor.split("_coord.csv")[0]
        if name not in human_samples:
            os.remove(os.path.join(base_dir, "Visium", "coord", coor))
    print("Pre-processing complete. Only human samples are retained.")

def find_common_genes(csv_files: str, min_genes=10000):
    """
    Find common genes in the given csv files.
    """
    save_path = os.path.join(os.path.dirname(os.path.dirname(csv_files[0])))
    print("Common genes will be saved to ", save_path)
    common_genes = None
    used_csvs = []
    current_gene_lists = []
    for file in tqdm(csv_files):
        with open(file, "r") as f:
            first_line = f.readline()
        genes = first_line.split(",")
        genes[0] = "Unnamed: 0"
        genes[-1] = genes[-1].split("\n")[0]
        genes = [x.split("hg38_")[-1] for x in genes]
        genes = set(genes)
        if len(genes) < min_genes:
            print("Sample has too few genes. Skipping")
            continue
        used_csvs.append(file)
        current_gene_lists.append(genes)
    # save common genes to a file
    current_gene_lists.sort(key=len)
    common_genes = set.intersection(*current_gene_lists)
    common_genes_L = sorted(list(common_genes))
    print(f"Processing finished. Total {len(common_genes_L)} genes in {len(used_csvs)} samples will be processed")
    used_csvs = sorted(used_csvs)
    used_csvs = [os.path.basename(sample).replace('_count.csv', '') for sample in used_csvs]
    with open(os.path.join(save_path, "sample_info", f"common_genes_{min_genes}.txt"), "w") as f:
        for gene in sorted(common_genes_L):
            f.write(f"{gene}\n")
    with open(os.path.join(save_path, "sample_info", f"samples_with_enough_genes_{min_genes}.txt"), "w") as f:
        for sample in sorted(used_csvs):
            f.write(f"{sample}\n")
    return common_genes_L, used_csvs

def process_sample(args):
    """
    Process a single sample to create patches and save them.
    """
    sample, base_dir, common_genes, name = args
    try:
        image = Image.open(os.path.join(base_dir, "Visium", "image", f"{sample}.png")).convert('RGB')
        exp_df = pd.read_csv(os.path.join(base_dir, "Visium", "gene_exp", f"{sample}_count.csv"))
        exp_df.columns = [x.split("hg38_")[-1] for x in exp_df.columns] #remove the genome reference
        exp_df = exp_df[common_genes]
        coor_df = pd.read_csv(os.path.join(base_dir, "Visium", "coord", f"{sample}_coord.csv"))
        patch_names = exp_df["Unnamed: 0"].tolist()
        coor_df = coor_df.drop(columns=["Unnamed: 0"])
        exp_df = exp_df.drop(columns=["Unnamed: 0"])
        for i, patch_name in enumerate(patch_names):
            patch = np.array([exp_df.iloc[i].values])
            # Normalize and log-2 transform the matx
            patch = patch / (np.sum(patch, axis=1, keepdims=True)+1e-10) * 1e6
            patch = np.log2(patch + 1)
            # get images
            x_centre, y_centre, radius = coor_df.iloc[i][["xaxis", "yaxis", "r"]]
            x = int(x_centre - radius)
            y = int(y_centre - radius)
            if x < 0 or y < 0 or x + 2 * radius > image.width or y + 2 * radius > image.height:
                print(f"Skipping patch {patch_name} for sample {sample} due to out of bounds coordinates.")
                continue
            patch_image = image.crop((x, y, x + 2 * radius, y + 2 * radius))
            if patch_image.size[0] * patch_image.size[1] < 10000:
                print("Image too small, skipping ...")
            else:
                # save image and numpy
                patch_image.save(os.path.join(base_dir, f"Visium_patch_images_filtered_processed_{name}", f"{patch_name}.png"))
                np.save(os.path.join(base_dir, f"Visium_patch_exps_filtered_processed_{name}", f"{patch_name}.npy"), patch)
    except Exception as e:
        print(f"Error processing sample {sample}: {e}")

def create_patch_matrix(base_dir: str, common_genes: list, sample_names: list, dir_name: str):
    """
    Create a matrix with only the common genes using multiprocessing.
    """
    if not sample_names:
        sample_names = [
            f.split(".")[0]
            for f in os.listdir(os.path.join(base_dir, "Visium", "image"))
            if f.endswith('.png')
        ]
    args = [(sample, base_dir, common_genes, dir_name) for sample in sample_names]

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print(f"Starting with {num_workers} processes")
    with get_context("spawn").Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_sample, args), total=len(sample_names), desc="Processing samples"))

    print("Patch matrix creation complete.")

def main():
    parser = argparse.ArgumentParser(description="Generate patch expression numpy file for STimage-1k4m dataset")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory of the STimage-1k4m dataset")
    parser.add_argument("--common_genes_txt", type=str, default=None, help="Path to the .txt file with all common genes")
    parser.add_argument("--used_samples", type=str, default=None, help="Path to the .txt file with all samples to be proccessed")
    parser.add_argument("--name", type=str, default=None, help="Name of the savind directories")
    args = parser.parse_args()

    # Image.MAX_IMAGE_PIXELS = 51150844200000
    Image.MAX_IMAGE_PIXELS = None

    base_dir = args.base_dir
    os.makedirs(os.path.join(base_dir, f"Visium_patch_images_filtered_processed_{args.name}"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, f"Visium_patch_exps_filtered_processed_{args.name}"), exist_ok=True)
    preprocess_stimage_1k4m(base_dir)

    if args.common_genes_txt and args.used_samples:
        with open(args.common_genes_txt, 'r') as f:
            common_genes = [line.strip() for line in f.readlines()]
        with open(args.used_samples, 'r') as f:
            used_samples = [line.strip() for line in f.readlines()]
        print(f"Common genes found: {len(common_genes)}")
    else:
        exp_files = [os.path.join(base_dir, "Visium", "gene_exp", f) for f in os.listdir(os.path.join(base_dir, "Visium", "gene_exp")) if f.endswith('.csv')]
        common_genes, used_samples = find_common_genes(exp_files)

    print("Start processing patches")    
    create_patch_matrix(base_dir, common_genes, used_samples, args.name)

if __name__ == "__main__":
    main()
