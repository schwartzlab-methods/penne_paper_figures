'''
Generate some plots using plt
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def plot_single_histogram(file_dirs, exp_name, output_dir):
    for idx, each in enumerate(file_dirs):
        data = np.load(each)
        plt.hist(data, bins=np.log2(data.size).astype(int)+1)
        plt.title(f"{exp_name}")
        plt.ylabel("Counts")
        plt.savefig(os.path.join(output_dir, f"{exp_name}_{idx}_single_histogram.png"))
        plt.close()

def plot_sbs_histogram(file_dirs, exp_name, output_dir):
    '''
    Plot side by side histograms by class
    Treat the first data in the list as the classes, the second as the data
    '''
    assert len(file_dirs) == 2, "There should be two files in the list"
    classes = np.load(file_dirs[0])
    data = np.load(file_dirs[1])
    unique_classes = np.unique(classes)
    plot_data = [data[classes == cls] for cls in unique_classes]
    plt.hist(plot_data, bins=10, label=unique_classes)
    plt.title(f"{exp_name}")
    plt.ylabel("Counts")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{exp_name}_side_by_side_his.png"))
    plt.close()

def plot_bar(file_dirs, exp_name, output_dir):
    for idx, each in enumerate(file_dirs):
        data = np.load(each)
        x = np.unique(data)
        y = [np.sum(data == cls) for cls in x]
        plt.bar(x, y)
        plt.title(f"{exp_name}")
        plt.ylabel("Counts")
        plt.savefig(os.path.join(output_dir, f"{exp_name}_{idx}_bar.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dirs', type=str, nargs="+", required=True,
                        help="The directories of the files to be plotted")
    parser.add_argument('--exp_name', type=str, default="experiment")
    parser.add_argument('--type', type=str, default="single_histogram", 
                        help="The type of plot to be generated")
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.type == "single_histogram":
        plot_single_histogram(args.file_dirs, args.exp_name, args.output_dir)
    elif args.type == "side_by_side_histogram":
        plot_sbs_histogram(args.file_dirs, args.exp_name, args.output_dir)
    elif args.type == "bar":
        plot_bar(args.file_dirs, args.exp_name, args.output_dir)

if __name__ == '__main__':
    main()

