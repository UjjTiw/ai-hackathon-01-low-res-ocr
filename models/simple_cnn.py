import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetCustom(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetCustom, self).__init__()
        
        # Load pretrained efficientnet-b4 instead of b7
        # B4 offers better balance of accuracy and speed
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        
        # Modify the stem to better handle grayscale input
        self.grayscale_conv = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.grayscale_conv.weight)
        
        # Add attention mechanism
        in_features = self.model._fc.in_features
        self.attention = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, in_features),
            nn.Sigmoid()
        )
        
        # Replace classifier with a sequence for better performance
        self.model._fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Handle grayscale input with dedicated conv layer
        if x.size(1) == 1:
            x = self.grayscale_conv(x)
        
        # Get features from base model
        features = self.model.extract_features(x)
        
        # Global pooling
        x = self.model._avg_pooling(features)
        x = x.flatten(start_dim=1)
        
        # Apply attention
        att = self.attention(x)
        x = x * att
        
        # Classification head
        x = self.model._dropout(x)
        x = self.model._fc(x)
        
        return x