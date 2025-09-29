'''
Compute the expression level of all genes in a matrix and a label numpy array
'''

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
import seaborn as sns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the gene expression matrix (numpy file)')
    parser.add_argument('--label_path', type=str, required=True, help='Path to the label array (numpy file)')
    parser.add_argument('--labels_to_compare', type=str, nargs='+', required=True, help='List of labels to compare')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the computed expression levels (numpy file)')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    data = np.load(args.data_path)  # Assuming shape (cells, genes)
    labels = np.load(args.label_path)  # Assuming shape (cells,)

    expre_df = pd.DataFrame(data, index=labels)

    # Compute the expression level for each label
    expression_levels = {}
    for label in args.labels_to_compare:
        expression_levels[label] = expre_df.loc[label].mean(axis=0)

    # plot a voilin plot for each comparison
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=[expre_df.loc[label].mean(axis=1) for label in args.labels_to_compare],
                   x=args.labels_to_compare)

    plt.title("Gene Expression Levels")
    plt.xlabel("Labels")
    plt.ylabel("Expression Level")
    plt.savefig(os.path.join(args.output_path, "expression_levels_violin_plot.png"))
    plt.close()


if __name__ == "__main__":
    main()