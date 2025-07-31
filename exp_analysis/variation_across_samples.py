# compute of variation of all genes across samples
#! todo: add the option to selection time frames (ie the files) from another npy array
import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expression_npy', type=str, nargs="+", help='Path to expression numpy matrix')
    parser.add_argument('--gene_name', type=str, help='Name of the gene to plot')
    parser.add_argument('--output', type=str, help='Path to the output directory')
    parser.add_argument('--exp_name', type=str, help="Experiment name for the heatmap")
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    exp_L = []
    for each in args.expression_npy:
        exp_L.append(np.load(each))
    expression_matrix = np.concatenate(exp_L)
    gene_names = np.load(args.gene_name)
    # revert the log2 transform in the prediction
    expression_matrix = np.clip(expression_matrix, 0, None)
    expression_matrix = 2**expression_matrix - 1
    # normalize to counts per million
    expression_matrix = expression_matrix / np.sum(expression_matrix, axis=1, keepdims=True) * 1e6
    # log2 transform
    expression_matrix = np.log2(expression_matrix + 1)

    # compute the variation across samples
    variation = np.std(expression_matrix, axis=0)
    variation_df = pd.DataFrame(variation, columns=['Variation'])
    variation_df['Gene'] = gene_names
    variation_df = variation_df.set_index('Gene')
    variation_df = variation_df.sort_values(by='Variation', ascending=False)
    plt.figure(figsize=(100, 6))
    sns.barplot(
        x=variation_df.index,
        y=variation_df['Variation'],
        palette='viridis'
    )
    plt.xticks(rotation=90)
    plt.title(f"Variation of {args.exp_name} across samples")
    plt.xlabel("Genes")
    plt.ylabel("Standard Deviation of Expression")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, f'variation_across_samples_{args.exp_name}.png'))
    plt.close()

    # wrtie most varied genes names to txt
    top_genes = variation_df.head(20).index.tolist()
    with open(os.path.join(args.output, f'top_20_most_varied_genes_{args.exp_name}.txt'), 'w') as f:
        for gene in top_genes:
            f.write(f"{gene}\n")
    print(f"Top 20 most varied genes written to {os.path.join(args.output, f'top_20_most_varied_genes_{args.exp_name}.txt')}")
    print(f"Variation across samples plot saved to {os.path.join(args.output, f'variation_across_samples_{args.exp_name}.png')}")

if __name__ == "__main__":
    main()