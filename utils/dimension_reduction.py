import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import argparse
import os
import altair as alt
from toomanycells import TooManyCells as tmc
import anndata as ad
## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

def plot_umap(data, labels, save_dir, exp_name, extractor,
              n_neighbors=5,min_dist=0.01,metric="cosine"):
    '''
    Plot the umap of the data
    '''
    reducer = umap.UMAP(n_neighbors=n_neighbors,min_dist=min_dist,metric=metric)
    embedding = reducer.fit_transform(data)
    plt.figure(figsize=(10, 10))
    # generate colours
    unique_classes = list(set(labels))
    colors = plt.cm.get_cmap('tab20c', len(unique_classes))
    class_to_color = {cls: colors(i) for i, cls in enumerate(unique_classes)}
    # plot
    for (x, y), cls in zip(embedding, labels):
        if cls == "Background":
            continue
        plt.scatter(x, y, color=class_to_color[cls], label=cls)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=class_to_color[cls], markersize=10, label=cls)
           for cls in unique_classes]
    plt.legend(handles=handles, title="Classes")
    # x and y labels
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.savefig(os.path.join(save_dir, f"umap_{exp_name}_{n_neighbors}_{extractor}.png"))
    plt.close()

def tmc_plot(data, labels, save_dir, use_neg_modularity=False):
    adata = ad.AnnData(X=data, obs=pd.DataFrame({"cell_type": labels.tolist()}))
    tmc_obj = tmc(adata, os.path.join(save_dir, "tmc_output"))
    if use_neg_modularity:
        tmc_obj.eps = -10000 # set the modularity to be negative
    tmc_obj.run_spectral_clustering()
    tmc_obj.store_outputs(
        cell_ann_col="cell_type",
    )

def prepare_data(data, labels):
    '''
    Concat data by labeling each data point with its label
    '''
    label_L = []
    data_L = []
    for i, each in enumerate(data):
        features = np.load(each)
        features = features.reshape(features.shape[0], -1)
        data_L.append(features)
        if os.path.isfile(labels[i]):
            label_L.append(np.load(labels[i]))
        else:
            length = features.shape[0]
            label_L.append(np.array([labels[i]] * length))
    label_array = np.concatenate(label_L, axis=None)
    data_array = np.concatenate(data_L, axis=0)
    return data_array, label_array

def main(path_1, path_label, save_dir, extractor_name, exp_name, do_tmc, do_negative_modularity):
    original, cell_type = prepare_data(path_1, path_label)

    print("Numpy files loaded")
    print("Feature Shape:", original.shape)
    print("Cell type Shape:",cell_type.shape)

    # map cell type names to numbers
    cell_type_dict = {}
    for i, cell in enumerate(np.unique(cell_type)):
        cell_type_dict[cell] = i
    cell_type_num = [cell_type_dict[cell] for cell in cell_type]

    # TMC
    if do_tmc:
        tmc_plot(original,cell_type,save_dir,do_negative_modularity)

    # pca
    pca = PCA(n_components=50)
    if extractor_name == "phikon-v2":
        embedding_original = pca.fit_transform(original[:,0,0,:])
    else:
        embedding_original = pca.fit_transform(original)
    var_ex_original = pca.explained_variance_ratio_

    # prep pandas for altair scatter
    df_original = pd.DataFrame({f"PC 1 ({var_ex_original[0]:.2f})": embedding_original[:, 0],
                                f"PC 2 ({var_ex_original[1]:.2f})": embedding_original[:, 1],
                                "Classes": cell_type})
    
    # plot with altair
    scatter1 = alt.Chart(df_original).mark_point().encode(
        x=f"PC 1 ({var_ex_original[0]:.2f})",
        y=f"PC 2 ({var_ex_original[1]:.2f})",
        color="Classes",
    )
    scatter1.interactive().save(os.path.join(save_dir, f"feature_pca_{extractor_name}_{exp_name}_original.html"))
    
    # plot with plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    scatter1 = ax.scatter(embedding_original[:, 0], embedding_original[:, 1], c=cell_type_num, cmap='tab10', s=0.1)
    ax.set_title(extractor_name)
    handles, labels = scatter1.legend_elements()
    ax.legend(handles, np.unique(cell_type), title="Class", loc='center left', bbox_to_anchor=(1.04, 0.5))
    ax.set_xlabel(f"PC 1 (Variance explained: {var_ex_original[0]:.2f})")
    ax.set_ylabel(f"PC 2 (Variance explained: {var_ex_original[1]:.2f})")
    final_save_dir = os.path.join(save_dir, f"feature_pca_{extractor_name}_{exp_name}.png")
    plt.savefig(final_save_dir, bbox_inches="tight")
    plt.close()

    # plot umap with plt
    if extractor_name == "phikon-v2":
        plot_umap(original[:,0,0,:], cell_type, save_dir, extractor_name, exp_name)
    else:
        plot_umap(original, cell_type, save_dir, extractor_name, exp_name)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Plotting dimension reduction")
    argparser.add_argument("--save_dir", type=str, help="The directory to save the plot")
    argparser.add_argument("--path", type=str, nargs="+", help="Paths to feature extractor features")
    argparser.add_argument("--path_label", type=str, nargs="+", help="Paths or names to the labels")
    argparser.add_argument("--do_tmc", action="store_true", help="Run TMC")
    argparser.add_argument("--do_negative_modularity", action="store_true", help="Run TMC with negative modularity")
    argparser.add_argument("--extractor", type=str, help="name of the feature extractor")
    argparser.add_argument("--exp_name", type=str, help="name of the experiment")
    args = argparser.parse_args()
    main(args.path, args.path_label, args.save_dir, args.extractor, args.exp_name, args.do_tmc, args.do_negative_modularity)

