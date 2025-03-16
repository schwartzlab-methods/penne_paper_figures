'''
Generate the feature representation of the tissue level using the attention maps of the model.
Instead of using the CLS token, we take the weights of CLS token and do a normalized weighted sum
'''

import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from _feature_extractors import owkin_features
from dataset import PanNukeDataset
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as v2
import argparse

def get_attn_features(extractor, processor, x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # convert x to tensor
    toimage = v2.ToImage()
    x = toimage(x)
    # x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0)
    # get attention and features
    attention, features = owkin_features(extractor, device, processor, x, return_attn=True)
    return attention, features

def weighted_sum_attention(attention: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    # average over the heads
    avg_attention = attention.mean(dim=1)
    cls_attn = avg_attention[:, 0, 1:]
    # normalize the attention
    normalized_cls_attn = cls_attn / cls_attn.sum(dim=-1, keepdim=True)
    # weighted sum, where each patch is weighted by the attention
    # sum together to get a final feature representation with size (batch_size, hidden_size)
    weighted_sum = torch.einsum("bh,bhs->bs", normalized_cls_attn, features)
    return weighted_sum
    
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
    output = []
    for idx, each in enumerate(tqdm(data)):
        attn, fea = get_attn_features(feature_extractor, image_processor, each[0])
        weighted_sum = weighted_sum_attention(attn, fea)
        output.append(weighted_sum.view(-1).detach().cpu().numpy())
    output = np.array(output)
    np.save(os.path.join(args.output_dir, "features_phikon-v2_avg_attn.npy"), output)


if __name__ == '__main__':
    main()