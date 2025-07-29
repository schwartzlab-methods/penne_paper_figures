import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os
import argparse

def compute_genes(args, if_same_direction=True):
    dfs = []
    name = "same_dir" if if_same_direction else "different_dir"
    for i, path in enumerate(args.csv):
        df = pd.read_csv(path, index_col=0)
        # df = df[df['adj_p_value'] < 0.05]  # filter by adjusted p-value
        if if_same_direction:
            df = df[df['correct_dir'].isin(["Same_significant", "Same_nonsignificant"])]
        else:
            df = df[df['correct_dir'].isin(["Different_significant", "Different_nonsignificant"])]
        print(f"Number of genes in dataset {i}:", len(df))
        dfs.append(df)

    setlist = [set(df.index) for df in dfs]
    common_genes = set.intersection(*setlist)
    print("Number of common genes across all cell types", len(np.unique(list(common_genes))))
    with open(os.path.join(args.output, f"common_genes_{name}.txt"), "w") as f:
        for gene in common_genes:
            f.write(gene + "\n")

    # compute correlation for each dataframe
    with open(os.path.join(args.output, f"gene_correlation_stats_{name}.txt"), "w") as f:
        f.write("==================ACROSS SAMPLES========================" + "\n")
        f.write("======COMMON GENES (GENES THAT ARE SAME DIRECTION IN ALL SAMPLES)=======" + "\n")
        for df in dfs:
            df = df.loc[list(common_genes)]  # filter to only have the common genes
            corr, _ = spearmanr(df['log_fc'], df['log_fc_gt'])
            f.write(f"Spearman correlation: {corr:.4f}" + "\n")

        f.write("======ALL GENES (sample direction in all samples)=======")
        for df in dfs:
            corr, _ = spearmanr(df['log_fc'], df['log_fc_gt'])
            f.write(f"Spearman correlation: {corr:.4f}" + "\n")

    # correlation across genes
    df_gene = pd.DataFrame({
        "gene": dfs[0].loc[list(common_genes)].index,
        "spearman_cross_samples": [spearmanr([dfs[i].loc[gene]['log_fc'] for i in range(len(dfs))],
                                            [dfs[i].loc[gene]['log_fc_gt'] for i in range(len(dfs))])[0] 
                                            for gene in dfs[0].loc[list(common_genes)].index] 
                                            
    })
    # plot histogram
    plt.hist(df_gene['spearman_cross_samples'], bins=50)
    plt.xlabel('Spearman correlation')
    plt.ylabel('Frequency')
    plt.title('Spearman correlation across genes')
    plt.savefig(os.path.join(args.output, f"spearman_per_gene_{name}.png"))
    plt.close()

    df_gene.to_csv(os.path.join(args.output, f"gene_wise_correlation_{name}.csv"))
    high_spearman_genes = df_gene[df_gene['spearman_cross_samples'] > 0.5]['gene'].tolist()
    print(f"Number of genes with high Spearman correlation: {len(high_spearman_genes)}")
    with open(os.path.join(args.output, f"high_spearman_genes_{name}.txt"), "w") as f:
        for gene in high_spearman_genes:
            f.write(gene + "\n")

def compute_genes_all(args):
    dfs = []
    name = "all"
    for i, path in enumerate(args.csv):
        df = pd.read_csv(path, index_col=0)
        # df = df[df['adj_p_value'] < 0.05]  # filter by adjusted p-value
        print(f"Number of genes in dataset {i}:", len(df))
        dfs.append(df)

    setlist = [set(df.index) for df in dfs]
    common_genes = set.intersection(*setlist)
    print("Number of common genes across all cell types", len(np.unique(list(common_genes))))
    with open(os.path.join(args.output, f"common_genes_{name}.txt"), "w") as f:
        for gene in common_genes:
            f.write(gene + "\n")

    # compute correlation for each dataframe
    with open(os.path.join(args.output, f"gene_correlation_stats_{name}.txt"), "w") as f:
        f.write("==================ACROSS SAMPLES========================" + "\n")
        for df in dfs:
            corr, _ = spearmanr(df['log_fc'], df['log_fc_gt'])
            f.write(f"Spearman correlation: {corr:.4f}" + "\n")

    # correlation across genes
    df_gene = pd.DataFrame({
        "gene": dfs[0].loc[list(common_genes)].index,
        "spearman_cross_samples": [spearmanr([dfs[i].loc[gene]['log_fc'] for i in range(len(dfs))],
                                            [dfs[i].loc[gene]['log_fc_gt'] for i in range(len(dfs))])[0] 
                                            for gene in dfs[0].loc[list(common_genes)].index] 
                                            
    })
    # plot histogram
    plt.hist(df_gene['spearman_cross_samples'], bins=50)
    plt.xlabel('Spearman correlation')
    plt.ylabel('Frequency')
    plt.title('Spearman correlation across genes')
    plt.savefig(os.path.join(args.output, f"spearman_per_gene_{name}.png"))
    plt.close()

    df_gene.to_csv(os.path.join(args.output, f"gene_wise_correlation_{name}.csv"))
    high_spearman_genes = df_gene[df_gene['spearman_cross_samples'] > 0.5]['gene'].tolist()
    print(f"Number of genes with high Spearman correlation: {len(high_spearman_genes)}")
    with open(os.path.join(args.output, f"high_spearman_genes_{name}.txt"), "w") as f:
        for gene in high_spearman_genes:
            f.write(gene + "\n")

def main():
    parser = argparse.ArgumentParser(description="Identify consistently accurately predicted genes")
    parser.add_argument('--output', type=str, required=True, help='Output directory for results')
    parser.add_argument('--csv', type=str, required=True, nargs="+", help='Paths csv containing DEG analysis')
    args = parser.parse_args()
    compute_genes(args, if_same_direction=True)
    compute_genes(args, if_same_direction=False)
    compute_genes_all(args)

if __name__ == "__main__":
    main()