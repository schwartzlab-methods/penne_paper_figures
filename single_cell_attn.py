'''
Generate the feature representation of all cells in the PanNuke dataset using the Phikon-v2 model.
To get each cell, we treat the top 60% of the attention map as the most important regions.
We then extract the features from the model for each of these regions by calculating an averged sum of the features using the attention map and embeddings.
'''

import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from _feature_extractors import owkin_features
from dataset import PanNukeDataset
import numpy as np
import os
import torchvision.transforms.v2 as v2
from scipy.ndimage import label
import argparse

def get_attn_features(extractor, processor, x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # convert x to tensor
    toimage = v2.ToImage()
    x = toimage(x).unsqueeze(0)
    # reshape the image to (3, 256, 256)
    resizer = v2.Resize(256)
    x = resizer(x)
    # centre crop to (224, 224)
    cropper = v2.CenterCrop(224)
    x = cropper(x)
    # x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0)
    # get attention and features
    attention, features = owkin_features(extractor, device, processor, x, return_attn=True)
    return x, attention, features

def compute_single_cell_features(attn: torch.Tensor, img: torch.Tensor, fea: torch.Tensor, 
                       size: int = 224, threshold: float = 0.6) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    '''
    Process the attention values to generate a heatmap of size x size based on threshold
    !!!! Only works if the batch size is 1 !!!!!
    Shapes:
    attn: (1, num_heads, num_patches, num_patches)
    img: (1, 3, 224, 224)
    fea: (1, num_patches+1, hidden_size)

    Returns:
    list of tuples, each tuple contains:
    - cell_img: np.ndarray of size (size, size, 3)
    - cell_mask: np.ndarray of size (size, size)
    - cell_weighted_feature: np.ndarray of size (hidden_size)
    '''
    #! step1: generate the tresholded attention map
    # get the attention except for the cls token
    class_token_attention = attn[:, :, 0, 1:]
    nh = class_token_attention.shape[1]
    # we keep only a certain percentage of the mass according to threshold
    val, idx = torch.sort(class_token_attention.reshape(nh, -1))
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(class_token_attention.shape[1]):
        th_attn[head] = th_attn[head][idx2[head]]
    num_patches_side = int(class_token_attention.shape[-1] ** 0.5)
    class_token_attention = th_attn.reshape(1, nh, num_patches_side, num_patches_side).float()
    # average all the head
    class_token_attention = class_token_attention.mean(dim=1) # (1, num_patches, num_patches)
    attention_map = class_token_attention.view(num_patches_side, num_patches_side).cpu() # (num_patches, num_patches)

    #! step2: segment the cells using the threshold
    # change the attention map to binary
    attention_map_binary = (attention_map > 0.2).int()
    # get the connected components
    labels, num_cells = label(attention_map_binary.numpy())
    print("Num cells identified: ", num_cells)
    
    #! step3: get the CLS attention for each cell
    reshaped_features = fea[0, 1:, :].cpu() # (num_patches, hidden_size)
    np_attention_map = attention_map.cpu().numpy() # (num_patches, num_patches)
    cell_img_L = []
    cell_mask_L = []
    cell_feature_L = []
    cell_size_L = []
    for cell in range(1, num_cells+1):
        cell_mask: np.ndarray = (labels == cell).astype(np.float32) # (num_patches**0.5, num_patches**0.5)
        cell_attn: np.ndarray = np_attention_map * cell_mask # (num_patches**0.5, num_patches**0.5)
        cell_attn = cell_attn.reshape(-1) # (num_patches)
        cell_attn_tensor = torch.tensor(cell_attn) # (num_patches)
        cell_features = reshaped_features * cell_attn_tensor.unsqueeze(-1) # (num_patches, hidden_size)
        cell_weighted_feature = cell_features.sum(dim=0) # (hidden_size)
        # extrapolate masks to size x size, with only 1 and 0
        resized_mask: torch.Tensor = torch.nn.functional.interpolate(torch.tensor(cell_mask).unsqueeze(0).unsqueeze(0).float(), 
                                                       size=(size, size), mode='nearest')
        # convert the resized_mask to binary
        resized_mask = (resized_mask > 0).float()
        cell_size_L.append(resized_mask.sum().item())
        # print("Size of Cell: ", resized_mask.sum().item(), "pixels")
        # get the image of the cell
        cell_img: torch.Tensor = img * resized_mask
        cell_img_L.append(cell_img.cpu().numpy().squeeze(0).transpose(1, 2, 0))
        cell_mask_L.append(resized_mask.squeeze(0).cpu().numpy().transpose(1, 2, 0))
        cell_feature_L.append(cell_weighted_feature.cpu().numpy())

    return cell_img_L, cell_mask_L, cell_feature_L, cell_size_L

def determine_cell_type(cell_mask: np.ndarray, img_mask: np.ndarray) -> tuple[str, float]:
    '''
    Determine the cell type based on the mask and the label, obtain the most frequent label in the mask
    cell_mask: np.ndarray of size (224, 224, 1)
    img_mask: np.ndarray of size (224, 224, 6) in one-hot encoding
    Also provide the percent frequencies for the label
    '''
    label_dic = {
        0: "Neoplastic", 1: "Inflammatory", 2: "Connective/Soft_tissue", 3: "Dead", 4: "Epithelial", 5: "Background"
    }
    img_non_onehot: np.ndarray = np.argmax(img_mask, axis=-1)
    cell_mask_bool = (cell_mask > 0).reshape((cell_mask.shape[0], cell_mask.shape[1]))
    roi = img_non_onehot[cell_mask_bool]
    roi_label: int = np.argmax(np.bincount(roi.flatten()))
    roi_label_freq = np.bincount(roi.flatten())[roi_label] / roi.flatten().size
    return (label_dic[roi_label], roi_label_freq)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    data = PanNukeDataset(args.root_dir)
    print("Dataset created")
    feature_extractor = AutoModel.from_pretrained("/fs01/home/richarddong/.cache/huggingface/hub/phikon-v2")
    image_processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    print("Phikon-v2 Loaded as Feature Extractor")
    cell_type_L = []
    cell_size_L = []
    cell_feature_L = []
    cell_type_freq_L = []
    tissue_type_L = []
    for each in tqdm(data):
        img, attn, fea = get_attn_features(feature_extractor, image_processor, each[0])
        cell_img, cell_mask, cell_feature, cell_size = compute_single_cell_features(attn, img, fea)
        cell_size_L.extend(cell_size)
        # center crop the label to (224,224)
        img_mask = each[1]
        img_mask_tensor = torch.tensor(img_mask).permute(2, 0, 1)
        cropped_mask_tensor = v2.CenterCrop(224)(img_mask_tensor.unsqueeze(0))
        cropped_mask = cropped_mask_tensor.squeeze(0).permute(1,2,0).numpy()
        for idx, _ in enumerate(cell_img):
            cell_type, cell_freq = determine_cell_type(cell_mask[idx], cropped_mask)
            cell_type_L.append(cell_type)
            cell_type_freq_L.append(cell_freq)
            cell_feature_L.append(cell_feature[idx])
            tissue_type_L.append(each[2])
    cell_type_L = np.array(cell_type_L)
    cell_type_freq_L = np.array(cell_type_freq_L)
    cell_feature_L = np.array(cell_feature_L)
    tissue_type_L = np.array(tissue_type_L)
    cell_size_L = np.array(cell_size_L)
    np.save(os.path.join(args.output_dir, "cell_type.npy"), cell_type_L)
    np.save(os.path.join(args.output_dir, "cell_size.npy"), cell_size_L)
    np.save(os.path.join(args.output_dir, "cell_type_freq.npy"), cell_type_freq_L)
    np.save(os.path.join(args.output_dir, "cell_feature.npy"), cell_feature_L)
    np.save(os.path.join(args.output_dir, "tissue_type.npy"), tissue_type_L)

if __name__ == '__main__':
    main()