import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Identify consistently accurately predicted genes")
    parser.add_argument('--output', type=str, required=True, help='Output directory for results')
    parser.add_argument('--csv', type=str, required=True, nargs="+", help='Paths csv containing DEG analysis')
    args = parser.parse_args()
    dfs = []
    for path in args.csv:
        df = pd.read_csv(path, index_col=0)
        # df = df[df['adj_p_value'] < 0.05]  # filter by adjusted p-value
        df = df[df['correct_dir'] == "Same"]
        print(len(df))
        dfs.append(df)

    setlist = [set(df.index) for df in dfs]
    common_genes = set.intersection(*setlist)
    print(len(np.unique(list(common_genes))))
    with open(os.path.join(args.output, "common_genes.txt"), "w") as f:
        for gene in common_genes:
            f.write(gene + "\n")

    # compute correlation for each dataframe
    print("==================ACROSS SAMPLES========================")
    print("======COMMON GENES (GENES THAT ARE SAME DIRECTION IN ALL SAMPLES)=======")
    for df in dfs:
        df = df.loc[list(common_genes)]  # filter to only have the common genes
        corr, _ = spearmanr(df['log_fc'], df['log_fc_gt'])
        print(f"Spearman correlation: {corr:.4f}")

    print("======ALL GENES (sample direction in all samples)=======")
    for df in dfs:
        corr, _ = spearmanr(df['log_fc'], df['log_fc_gt'])
        print(f"Spearman correlation: {corr:.4f}")

    # correlation across genes
    import matplotlib.pyplot as plt
    df_gene = pd.DataFrame({
        "gene": df.loc[list(common_genes)].index,
        "spearman_cross_samples": [spearmanr([dfs[i].loc[gene]['log_fc'] for i in range(len(dfs))],
                                            [dfs[i].loc[gene]['log_fc_gt'] for i in range(len(dfs))])[0] 
                                            for gene in df.loc[list(common_genes)].index] 
                                            
    })
    # plot histogram
    plt.hist(df_gene['spearman_cross_samples'], bins=50)
    plt.xlabel('Spearman correlation')
    plt.ylabel('Frequency')
    plt.title('Spearman correlation across genes')
    plt.savefig(os.path.join(args.output, "spearman_per_gene.png"))

    df_gene.to_csv(os.path.join(args.output, "gene_wise_correlation.png"))
    high_spearman_genes = df_gene[df_gene['spearman_cross_samples'] > 0.5]['gene'].tolist()
    print(f"Number of genes with high Spearman correlation: {len(high_spearman_genes)}")
    with open(os.path.join(args.output, "high_spearman_genes.txt"), "w") as f:
        for gene in high_spearman_genes:
            f.write(gene + "\n")

if __name__ == "__main__":
    main()