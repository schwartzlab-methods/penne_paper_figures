'''
Feature extractors for the data
Currently supports: Phikon-v2, ResNet50 (ImageNet)
'''

import torch
import torchvision.transforms.v2 as v2

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
    
def resnet_features(model, device, x):
    model.to(device)
    model.eval()
    with torch.no_grad():
        toimage = v2.ToImage()
        x = toimage(x)
        # x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0)
        todytpe = v2.ToDtype(torch.float32, scale=True)
        x = todytpe(x)
        x = torch.clamp(x, max=1, min=0) #correct for float overflow
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