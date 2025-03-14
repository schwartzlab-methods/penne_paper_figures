'''
Visualize the attention maps of the model using Phikon-v2 on the PanNuke dataset.
'''

import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from _feature_extractors import owkin_features
from dataset import PanNukeDataset
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

def generate_attn_maps(attn: torch.Tensor, size: int = 256, 
                       threshold: float = 0.6) -> np.ndarray:
    '''
    Process the attention values to generate a heatmap of size x size based on threshold
    '''
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
    class_token_attention = class_token_attention.mean(dim=1)
    attention_map = class_token_attention.view(num_patches_side, num_patches_side).cpu()
    # extrapolate to size x size
    resized_attn = torch.nn.functional.interpolate(attention_map.unsqueeze(0).unsqueeze(0), size=(size, size), 
                                                        mode='bicubic')
    # return a numpy array of size x size (no batch or channel)
    return resized_attn.squeeze(0).squeeze(0).cpu().numpy()

def get_attn_features(extractor, processor, x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # convert x to tensor
    x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0)
    # get attention and features
    attention, _ = owkin_features(extractor, device, processor, x, return_attn=True)
    return attention

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    data = PanNukeDataset(args.root_dir)
    feature_extractor = AutoModel.from_pretrained("/fs01/home/richarddong/.cache/huggingface/hub/phikon-v2")
    image_processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    for idx, each in enumerate(tqdm(data)):
        attn = get_attn_features(feature_extractor, image_processor, each[0])
        resized_attn = generate_attn_maps(attn)
        # save the attention map
        img = each[0] / 255
        gray_image = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
        plt.imshow(gray_image, cmap='gray')
        plt.imshow(resized_attn, alpha=0.6, cmap='plasma')
        plt.colorbar()
        plt.savefig(os.path.join(args.output_dir, f"{idx}.png"))
        plt.close()
        
if __name__ == '__main__':
    main()