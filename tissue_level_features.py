import torch
from transformers import AutoImageProcessor, AutoModel
from torchvision.models import resnet50
from _feature_extractors import owkin_features, resnet_features
from dataset import PanNukeDataset
import torchvision.transforms.v2 as v2
import numpy as np
import os
from tqdm import tqdm
import argparse

def get_attn_features(extractor, processor, x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # convert x to image tensor
    toimage = v2.ToImage()
    x = toimage(x)
    # x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0)
    # get attention and features
    attention, features = owkin_features(extractor, device, processor, x, return_attn=True)
    return attention, features

def get_features(extractor, processor, x):
    attention, features = get_attn_features(extractor, processor, x[0])
    return attention, features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--extractor', type=str, default="phikon-v2")
    args = parser.parse_args()
    data = PanNukeDataset(args.root_dir)
    feature_L = []
    label_L = []
    if args.extractor == "phikon-v2":
        feature_extractor = AutoModel.from_pretrained("/fs01/home/richarddong/.cache/huggingface/hub/phikon-v2")
        image_processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        for each in tqdm(data):
            _, features = get_features(feature_extractor, image_processor, each)
            feature_L.append(features.cpu().detach().numpy())
            label_L.append(each[2])
    else:
        feature_extractor = resnet50(weights='DEFAULT')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for each in tqdm(data):
            features = resnet_features(feature_extractor, device, each[0])
            feature_L.append(features.view(-1).cpu().detach().numpy())
            label_L.append(each[2])
    np.save(os.path.join(args.output_dir, f'features_{args.extractor}.npy'), np.array(feature_L))
    np.save(os.path.join(args.output_dir, f'labels_{args.extractor}.npy'), np.array(label_L))
    

if __name__ == '__main__':
    main()