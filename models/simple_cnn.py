import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetCustom(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetCustom, self).__init__()
        
        # Load pretrained efficientnet-b7
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        
        # Convert grayscale to RGB approach
        # No need to modify the input layer, we'll handle this in forward
        
        # Replace classifier head
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Convert grayscale to RGB by repeating the channel
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)