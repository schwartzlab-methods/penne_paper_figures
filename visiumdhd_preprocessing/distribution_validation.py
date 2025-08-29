'''
Plot the cumulative distribution of gene expressions wrt total gene exp per cell
'''
from cmapPy.pandasGEXpress.parse_gct import parse
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
    - data: numpy array of shape (obs x genes)
    - labels: label for each obs
    - save_dir: directory to save the plots
    - exp_name: name of the experiment for saving the plot
    """

    cdf_dic = {}
    print("Calculating CDF data...")
    for i, label in enumerate(tqdm(labels)):
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
    
    mean_cdf_dic = pd.DataFrame(mean_cdf)
    mean_cdf_dic.to_csv(os.path.join(save_dir, f"mean_cdf_{exp_name}.csv"), index=False)
    std_cdf_dic = pd.DataFrame(std_cdf)
    std_cdf_dic.to_csv(os.path.join(save_dir, f"std_cdf_{exp_name}.csv"), index=False)

    unique_labels = list(mean_cdf.keys())

    # Plotting
    plt.figure(figsize=(10, 6))
    for label in unique_labels:
        plt.plot(mean_cdf[label], label=label)
        plt.fill_between(range(len(mean_cdf[label])),
                         mean_cdf[label] - std_cdf[label],
                         mean_cdf[label] + std_cdf[label],
                         alpha=0.2)
    plt.title("Cumulative Distribution of Gene Expression")
    plt.xlabel("Genes (sorted by expression)")
    plt.ylabel("Cumulative Distribution Function (CDF)")
    plt.legend(title="Sample")
    plt.savefig(os.path.join(save_dir, f"cdf_plot_{exp_name}.png"))
    plt.close()
    print("Figure saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_npy', type=str, required=True, nargs="+", help='Path to the numpy file containing the gene expression data')
    parser.add_argument('--sample_names', type=str, nargs="+", default=None, help="Labels for each data npy")
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the plots')
    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment for saving the plot')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    data_L = []
    sample_L = []
    # Load the data
    for idx, path in enumerate(args.data_npy):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file {path} does not exist.")
        print(f"Loading from {path}")
        if path.endswith(".npy"):
            data = np.load(path)
            data_L.append(data)
            if args.sample_names:
                sample_L.extend([args.sample_names[idx]] * data.shape[0])
            else:
                sample_name = os.path.basename(os.path.dirname(path))
                sample_L.append([sample_name] * data.shape[0])
        elif path.endswith(".gct"): #if it is a gct file
            data = parse(path).data_df.to_numpy()
            data_L.append(data)
            if args.sample_names:
                sample_L.extend([args.sample_names[idx]] * data.shape[0])
            else:
                sample_name = os.path.basename(os.path.dirname(path))
                sample_L.append([sample_name] * data.shape[0])
        else:
            for file in tqdm(os.listdir(path)):
                if file.endswith('.npy'):
                    data = np.load(os.path.join(path, file))
                    data_L.append(data)
                    if args.sample_names:
                        sample_L.append(args.sample_names[idx])
                    else:
                        # sample name is the third last part of the direcotry structure
                        sample_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
                        sample_L.append(sample_name)
                elif file.endswith('.gct'):
                    data = parse(os.path.join(path, file)).data_df.to_numpy()
                    data_L.append(data)
                    if args.sample_names:
                        sample_L.append(args.sample_names[idx])
                    else:
                        # sample name is the third last part of the direcotry structure
                        sample_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
                        sample_L.append(sample_name)
    
    # fill 0 for entries that are not in the same shape
    print("Concating data...")
    max_shape = max(data.shape[1] for data in data_L)
    for i in tqdm(range(len(data_L))):
        if data_L[i].shape[1] < max_shape:
            padding = np.zeros((data_L[i].shape[0], max_shape - data_L[i].shape[1]))
            data_L[i] = np.concatenate((data_L[i], padding), axis=1)
    # Concatenate all data
    data = np.concatenate(data_L, axis=0)
    assert len(sample_L) == data.shape[0], f"Found shape {len(sample_L)} and {data.shape[0]}"

    # Plot the distribution
    print("Plotting CDF...")
    plot_distribution(data, sample_L, args.save_dir, args.exp_name)

if __name__ == "__main__":
    main()