'''
Plot the cumulative distribution of gene expressions wrt total gene exp per cell
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm


def plot_distribution(data, labels, save_dir, exp_name):
    """
    Plot the cumulative distribution of gene expressions wrt total gene exp per cell.

    The plot shows the mean cumulative distribution function (CDF) of gene expression
    with standard deviation bands for each label. 
    
    Parameters:
    - data: numpy array of shape (cells x genes)
    - save_dir: directory to save the plots
    - exp_name: name of the experiment for saving the plot
    """

    # normalize to 10e6 and log2 transform the data
    data = data / np.sum(data, axis=1, keepdims=True) * 1e6
    data = np.log2(data + 1)

    cdf_dic = {}
    for i, label in enumerate(labels):
        x = data[i, :].flatten()
        x_sorted = np.sort(x)[::-1]  # sort in descending order
        cdf = np.cumsum(x_sorted) / np.sum(x_sorted)  # cumulative distribution function
        if label not in cdf_dic:
            cdf_dic[label] = []
        cdf_dic[label].append(cdf)
    
    # calculate mean and std for each label
    mean_cdf = {}
    std_cdf = {}
    for label, cdfs in cdf_dic.items():
        mean_cdf[label] = np.mean(cdfs, axis=0)
        std_cdf[label] = np.std(cdfs, axis=0)
    
    unique_labels = list(cdf_dic.keys())
    df = pd.DataFrame({
        "Mean CDF": [mean_cdf[label] for label in unique_labels],
        "Std CDF": [std_cdf[label] for label in unique_labels],
        "Label": unique_labels
    })
    df.to_csv(os.path.join(save_dir, f"cdf_data_{exp_name}.csv"), index=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    for label in unique_labels:
        plt.plot(mean_cdf[label], label=label)
        plt.fill_between(range(len(mean_cdf[label])),
                         mean_cdf[label] - std_cdf[label],
                         mean_cdf[label] + std_cdf[label],
                         alpha=0.2)
    plt.title(f"Cumulative Distribution of Gene Expression - {exp_name}")
    plt.xlabel("Genes (sorted by expression)")
    plt.ylabel("Cumulative Distribution Function (CDF)")
    plt.legend(title="Visium Sample")
    plt.savefig(os.path.join(save_dir, f"cdf_plot_{exp_name}.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_npy', type=str, required=True, nargs="+", help='Path to the numpy file containing the gene expression data')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the plots')
    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment for saving the plot')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    data_L = []
    sample_L = []
    # Load the data
    for path in args.data_npy:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file {path} does not exist.")
        for file in tqdm(os.listdir(path), desc=f"Loading data from {path}"):
            if file.endswith('.npy'):
                data = np.load(os.path.join(path, file))
                data_L.append(data)
                # sample name is the third last part of the direcotry structure
                sample_name = os.path.dirname(os.path.dirname(os.path.dirname(path))) 
                sample_L.append(sample_name)
    # Concatenate all data
    data = np.concatenate(data_L, axis=0)

    # Plot the distribution
    plot_distribution(data, sample_L, args.save_dir, args.exp_name)