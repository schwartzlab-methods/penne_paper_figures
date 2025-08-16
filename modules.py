'''
Modules for the deep learning model
'''
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
import torch


#! Domain adaptation modules
#! These translate the feature vectors from both the real H&E and synthetic H&E images 
#! into the same space.
#! It uses a Domain Adversarial Neural Network (DANN) approach to align the feature distributions
#! of the two domains.

#* Translator for features
class Translator(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=512, output_dim=1024):
        super(Translator, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim) 
        self.bn = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))
        return self.fc2(x)  # No activation to keep range flexible

#* Gradient reversal layer (for adversarial training)
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DomainDiscriminator(nn.Module):
    def __init__(self, feature_in=1024, alpha=1.0, do_reversal=True):
        super(DomainDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_in, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.alpha = alpha
        self.do_reversal = do_reversal

    def forward(self, x):
        if self.do_reversal:
            out = GradReverse.apply(x, self.alpha)
            return self.model(out)
        else:
            return self.model(x)

#* Orthogonality translator
class OrthogonalTranslator(nn.Module):
    def __init__(self, feature_in=1024, feature_out=1024):
        super(OrthogonalTranslator, self).__init__()
        self.fc1 = nn.Linear(feature_in, feature_out)
        self.fc2 = nn.Linear(feature_out, feature_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

#! Prediction modules
#! Predict the whole transcriptome from the image
class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, p=0.2):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p) # Dropout layer to prevent overfitting

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return x

class GatedMLPBlock(nn.Module):
    def __init__(self, input_size, hidden_size, p=0.2):
        super(GatedMLPBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.gate = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(p) # Dropout layer to prevent overfitting

    def forward(self, x):
        gate = self.gate(x)
        x_proj = self.fc2(self.dropout(F.gelu(self.fc1(x))))
        return self.norm(x + gate * x_proj)

class PredictorGMLP(nn.Module):
    # Predicts the whole transcriptome from the image using a Gated MLP
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedMLPBlock(input_size, hidden_size, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(input_size, output_size) #make gene expression prediction

    def forward(self, x, return_gate = False):
        for layer in self.layers:
            x = layer(x)
        if return_gate:
            return x
        else:
            return self.output_proj(x)  # Final gene expression vector

#! modules for SPAGHETTI
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # padding, keep the image size constant after next conv2d
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

class SpaghettiGenerator(nn.Module):
    def __init__(self, in_channels, num_residual_blocks=9):
        super(SpaghettiGenerator, self).__init__()

        self.normalization = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # Inital Convolution  3*256*256 -> 64*256*256
        out_channels=64
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(in_channels), # padding, keep the image size constant after next conv2d
            nn.Conv2d(in_channels, out_channels, 2*in_channels+1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        channels = out_channels
        
        # Downsampling   64*256*256 -> 128*128*128 -> 256*64*64
        self.down = []
        for _ in range(2):
            out_channels = channels * 2
            self.down += [
                nn.Conv2d(channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            channels = out_channels
        self.down = nn.Sequential(*self.down)
        
        # Transformation (ResNet)  256*64*64
        self.trans = [ResidualBlock(channels) for _ in range(num_residual_blocks)]
        self.trans = nn.Sequential(*self.trans)
        
        # Upsampling  256*64*64 -> 128*128*128 -> 64*256*256
        self.up = []
        for _ in range(2):
            out_channels = channels // 2
            self.up += [
                nn.Upsample(scale_factor=2), # bilinear interpolation
                nn.Conv2d(channels, out_channels, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            channels = out_channels
        self.up = nn.Sequential(*self.up)
        
        # Out layer  64*256*256 -> 3*256*256
        self.out = nn.Sequential(
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(channels, in_channels, 2*in_channels+1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.normalization(x)
        x = self.conv(x)
        x = self.down(x)
        x = self.trans(x)
        x = self.up(x)
        x = self.out(x)
        # normalize to range [0,1]
        x = torch.clamp(x, min=-1, max=1)
        min_val = x.min()
        max_val = x.max()
        x = (x-min_val)/(max(max_val-min_val, 1e-5))
        x = torch.clamp(x, min=0, max=1) # ensure no overflow
        return x

#! PCM specific modules for aligning generating pcm predictions
class CellTypeClassifier(nn.Module):
    '''
    Classify the cell types from gene expression features.
    This is used to guide the training of the predictor to better at predicting the gene expression of the pcm
    This is a simple feedforward neural network with two fully connected layers. 
    No softmax activation is applied since it is used in a loss function that applies softmax internally (e.g., CrossEntropyLoss).
    '''
    def __init__(self, input_size, num_classes, hidden_size = 512):
        super(CellTypeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)  # Dropout layer to prevent overfitting
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x