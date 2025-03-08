import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import argparse
import os
import altair as alt
## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

def plot_umap(data, labels, save_dir, exp_name,
              n_neighbors=50,metric="cosine"):
    '''
    Plot the umap of the data
    '''
    reducer = umap.UMAP(n_neighbors=n_neighbors, metric=metric)
    embedding = reducer.fit_transform(data)
    plt.figure(figsize=(10, 10))
    # generate colours
    unique_classes = list(set(labels))
    colors = plt.cm.get_cmap('tab20c', len(unique_classes))  # Using 'tab10' for distinct colors
    class_to_color = {cls: colors(i) for i, cls in enumerate(unique_classes)}
    # plot
    for (x, y), cls in zip(embedding, labels):
        plt.scatter(x, y, color=class_to_color[cls], label=cls, edgecolors='black')
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=class_to_color[cls], markersize=10, label=cls)
           for cls in unique_classes]
    plt.legend(handles=handles, title="Classes")
    plt.savefig(os.path.join(save_dir, f"umap_{exp_name}_{n_neighbors}.png"))
    plt.close()

def main(path_1, path_label, save_dir, extractor_name, exp_name):
    original = np.load(path_1)
    cell_type = np.load(path_label)

    print("Numpy files loaded")
    print(original[:,0,0,:].shape)
    print(cell_type.shape)

    # map cell type names to numbers
    cell_type_dict = {}
    for i, cell in enumerate(np.unique(cell_type)):
        cell_type_dict[cell] = i
    cell_type_num = [cell_type_dict[cell] for cell in cell_type]

    # pca
    # pca = PCA(n_components=2)
    pca = PCA(n_components=50)
    embedding_original = pca.fit_transform(original[:,0,0,:])
    var_ex_original = pca.explained_variance_ratio_

    # prep pandas for altair scatter
    df_original = pd.DataFrame({f"PC 1 ({var_ex_original[0] * 100:.2f}%)": embedding_original[:, 0],
                                f"PC 2 ({var_ex_original[1] * 100:.2f}%)": embedding_original[:, 1],
                                "Classes": cell_type})
    
    # plot with altair
    scatter1 = alt.Chart(df_original).mark_point().encode(
        x=f"PC 1 ({var_ex_original[0] * 100:.2f}%)",
        y=f"PC 2 ({var_ex_original[1] * 100:.2f}%)",
        color="Classes",
    )
    scatter1.interactive().save(os.path.join(save_dir, f"feature_pca_{extractor_name}_{exp_name}_original.html"))
    
    # plot with plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    scatter1 = ax.scatter(embedding_original[:, 0], embedding_original[:, 1], c=cell_type_num, cmap='tab10', s=0.1)
    ax.set_title(extractor_name)
    # scatter2 = ax[1].scatter(embedding_spaghetti[:, 0], embedding_spaghetti[:, 1], c=cell_type_num, cmap='tab10', s=0.1)
    # ax[1].set_title(f"Spaghetti + {extractor_name}")
    handles, labels = scatter1.legend_elements()
    ax.legend(handles, np.unique(cell_type), title="Class", loc='center left', bbox_to_anchor=(1.04, 0.5))
    final_save_dir = os.path.join(save_dir, f"feature_pca_{extractor_name}_{exp_name}.png")
    plt.savefig(final_save_dir, bbox_inches="tight")
    plt.close

    #plot umap
    plot_umap(original[:,0,0,:], cell_type, save_dir, exp_name)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Plotting PCA")
    argparser.add_argument("--save_dir", type=str, help="The directory to save the plot")
    argparser.add_argument("--path", type=str, help="Path to feature extractor features")
    argparser.add_argument("--path_label", type=str, help="Path to the labels")
    argparser.add_argument("--extractor", type=str, help="name of the feature extractor")
    argparser.add_argument("--exp_name", type=str, help="name of the experiment")
    args = argparser.parse_args()
    main(args.path, args.path_label, args.save_dir, args.extractor, args.exp_name)

