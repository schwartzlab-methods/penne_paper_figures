'''
Generate pre-processing statistics for the Visium dataset.
'''
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Generate pre-processing statistics for the Visium dataset.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input data files.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output statistics file.')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    files = os.listdir(args.input)
    data_L = [np.load(os.path.join(args.input, each)) for each in tqdm(files) if each.endswith('.npy')]
    data = np.concatenate(data_L, axis=0)

    #! plot violin plots for gene (average across samples)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

    plt.figure(figsize=(10, 6))
    plt.violinplot([mean, std, min_val, max_val], showmedians=True, quantiles=[[0.25, 0.75]]*4)
    plt.xticks([1, 2, 3, 4], ['Mean', 'Std', 'Min', 'Max'])
    plt.title('Pre-processing Statistics Per Gene')
    plt.savefig(os.path.join(args.output, 'pre_processing_stats_log2normalized_gene.png'))
    plt.close()

    # find genes where max_val = 0
    zero_max_gene = np.where(max_val == 0)[0]
    if zero_max_gene.size > 0:
        print(f"Genes with max expression 0 across all samples:")
        for gene in zero_max_gene:
            print(f" - idx {gene}")
    
    # plot violin plots for the number of features expressed per sample
    num_features = np.sum(data > 0, axis=1)
    plt.figure(figsize=(10, 6))
    plt.violinplot(num_features, showmedians=True, quantiles=[[0.25, 0.75]])
    plt.title('Number of Features Expressed Per Sample')
    plt.savefig(os.path.join(args.output, 'num_features_expressed_per_sample.png'))
    plt.close()

    #! plot violin plots for samples (average across genes)
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    min_val = np.min(data, axis=1)
    max_val = np.max(data, axis=1)

    plt.figure(figsize=(10, 6))
    plt.violinplot([mean, std, min_val, max_val], showmedians=True, quantiles=[[0.25, 0.75]]*4)
    plt.xticks([1, 2, 3, 4], ['Mean', 'Std', 'Min', 'Max'])
    plt.title('Pre-processing Statistics Per Sample')
    plt.savefig(os.path.join(args.output, 'pre_processing_stats_log2normalized_sample.png'))
    plt.close()

    # find genes where max_val = 0
    zero_max_sample = np.where(max_val == 0)[0]
    if zero_max_sample.size > 0:
        print(f"Samples with max expression 0 across all genes:")
        for sample in zero_max_sample:
            print(f" - {files[sample]}")


if __name__ == "__main__":
    main()