'''
Modules for the deep learning model
'''
from torch import nn
import torch.nn.functional as F
import torch
from spaghetti._spahgetti_modules import GeneratorResNet


#! Domain adaptation modules
#! These translate the feature vectors from both the real H&E and synthetic H&E images 
#! into the same space.
#! It uses a Domain Adversarial Neural Network (DANN) approach to align the feature distributions
#! of the two domains.

# Translator for features
class Translator(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=512, output_dim=1024):
        super(Translator, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim) 
        self.bn = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))
        return self.fc2(x)  # No activation to keep range flexible

# Gradient reversal layer (for adversarial training)
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DomainDiscriminator(nn.Module):
    def __init__(self, alpha=1.0):
        super(DomainDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
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

    def forward(self, vec1, vec2):
        # Concatenate the two feature vectors
        vec1 = GradReverse.apply(vec1, alpha = self.alpha)
        vec2 = GradReverse.apply(vec2, alpha = self.alpha)
        # Concatenate the two feature vectors
        x = torch.cat((vec1, vec2), dim=1)
        return self.model(x)

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