import torchvision.transforms.v2 as v2
import torch
from transformers import AutoImageProcessor, AutoModel
from dataset import PanNukeDataset
import numpy as np
import os
from tqdm import tqdm
import argparse

def owkin_features(model, device, image_processor, x, return_attn = False):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        todytpe = v2.ToDtype(torch.float32, scale=True)
        x = todytpe(x)
        x = torch.clamp(x, max=1, min=0) #correct for float overflow
        inputs = image_processor(x, return_tensors="pt", do_rescale=False)
        outputs = model(**inputs.to(device),output_attentions=return_attn)
        extracted = outputs.last_hidden_state#[:, 0, :]
    # return last layer attention and full embedding
    # shape of attention is (batch_size, num_heads, seq_length, seq_length)
    # shape of extracted is (batch_size, seq_length, hidden_size)
    if return_attn: 
        return outputs.attentions[-1], extracted
    else:
        return extracted

def get_attn_features(extractor, processor, x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # convert x to tensor
    x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0)
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
    args = parser.parse_args()
    data = PanNukeDataset(args.root_dir)
    feature_extractor = AutoModel.from_pretrained("/fs01/home/richarddong/.cache/huggingface/hub/phikon-v2")
    image_processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    feature_L = []
    label_L = []
    for each in tqdm(data):
        _, features = get_features(feature_extractor, image_processor, each)
        feature_L.append(features)
        label_L.append(each[2])
    np.save(os.path.join(args.output_dir, 'features.npy'), np.array(feature_L))
    np.save(os.path.join(args.output_dir, 'labels.npy'), np.array(label_L))
    

if __name__ == '__main__':
    main()