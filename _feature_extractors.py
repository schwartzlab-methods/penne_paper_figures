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
        todytpe = v2.ToDtype(torch.float32)#, scale=True)
        x = todytpe(x)
        inputs = image_processor(x, return_tensors="pt", do_rescale=False)
        outputs = model(**inputs.to(device),output_attentions=return_attn)
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
