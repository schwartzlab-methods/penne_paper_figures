'''
Feature extractors for the data
Currently supports: Phikon-v2, ResNet50 (ImageNet)
'''

import torch
import torchvision.transforms.v2 as v2
import torch.nn.functional as F
from modules import SpaghettiGenerator
from transformers import AutoImageProcessor

def init_spaghetti(model_path: str) -> torch.nn.Module:
    '''
    Initialize the SPAGHETTI model for image translation
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = SpaghettiGenerator(3, 9)
    generator.to(device)
    ckpt = torch.load(model_path, map_location=device)["state_dict"]
    # get only G_AB weights
    ckpt = {k[5:]: v for k, v in ckpt.items() if ("G_AB" in k)}
    generator.load_state_dict(ckpt)
    return generator

#! Pre-processing for Phikon

def pre_processing_phikon(model=None):
    class ResizeShortestEdge:
        def __init__(self, size, interpolation=v2.InterpolationMode.BICUBIC):
            self.size = size
            self.interpolation = interpolation

        def __call__(self, img: torch.Tensor):
            # img is (C, H, W)
            if img.dim() != 3:
                raise ValueError(f"Expected (C, H, W) input, but got {img.shape}")

            c, h, w = img.shape
            if h < w:
                new_h = self.size
                new_w = int(w * self.size / h)
            else:
                new_w = self.size
                new_h = int(h * self.size / w)

            img = img.unsqueeze(0)  # add batch dimension: (1, C, H, W)
            img = F.interpolate(
                img,
                size=(new_h, new_w),
                mode=self.interpolation.value.lower(),
                align_corners=False if self.interpolation in [v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC] else None
            )
            return img.squeeze(0)  # back to (C, H, W)
    if model:
        image_processor = AutoImageProcessor.from_pretrained(model, use_fast=True)
        return lambda x: image_processor(x, return_tensors="pt", do_rescale=False)["pixel_values"]
    else:
        IMAGE_MEAN = [0.485, 0.456, 0.406]
        IMAGE_STD = [0.229, 0.224, 0.225]
        RESCALE_FACTOR = 0.00392156862745098  # = 1/255
        TARGET_SIZE = 224  # both resize shortest edge and crop
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            ResizeShortestEdge(TARGET_SIZE),                         # Resize shortest edge to 224
            v2.CenterCrop(TARGET_SIZE),                               # Center crop to 224x224
            v2.Lambda(lambda x: x * RESCALE_FACTOR),          # Rescale (1/255)
            v2.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),     # Normalize
        ])
        return lambda batch: torch.stack([transform(x) for x in batch])

def owkin_features(model, device, image_processor, x, return_attn = False, return_embedding = False):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        todytpe = v2.ToDtype(torch.float32)#, scale=True)
        x = todytpe(x)
        inputs = image_processor(x, return_tensors="pt", do_rescale=False)
        outputs = model(**inputs.to(device),output_attentions=return_attn,output_hidden_states=return_embedding)
    if return_embedding: # return the pathch embeddings
        return outputs.hidden_states[0][:,1:,:]
    # return last layer attention and full embedding
    # shape of attention is (batch_size, num_heads, seq_length, seq_length)
    # shape of extracted is (batch_size, seq_length, hidden_size)
    if return_attn: 
        return outputs.attentions[-1], outputs.last_hidden_state#[:, 0, :]
    else:
        return outputs.last_hidden_state#[:, 0, :]
    
def resnet_features(model, device, x):
    model.to(device)
    model.eval()
    with torch.no_grad():
        toimage = v2.ToImage()
        x = toimage(x)
        todytpe = v2.ToDtype(torch.float32)
        x = todytpe(x)
        x = x / 255
        preprocess = v2.Compose([
            v2.Resize(256),  
            v2.CenterCrop(224),  
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
        x = preprocess(x).unsqueeze(0)
        x = x.to(device)
        # get the features
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
    return x

def spaghetti_convertion(model, device, x):
    '''
    Perform the image translation of PCM -> HE using the SPAGHETTI model
    '''
    model.to(device)
    model.eval()
    with torch.no_grad():
        normalization = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        x = normalization(x)
        x = x.to(device)
        transformed = model(x)
        # normalize to range [0,1]
        out = torch.clamp(transformed, min=-1, max=1)
        min_val = out.min()
        max_val = out.max()
        out = (out-min_val)/(max(max_val-min_val, 1e-5))
        out = torch.clamp(out, min=0, max=1) # ensure no overflow
        
    return out
