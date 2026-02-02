import torch
import torch.nn as nn
import timm

class DFER_GCVIT(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(DFER_GCVIT, self).__init__()

        self.backbone = timm.create_model('gcvit_tiny', pretrained=pretrained, num_classes=0)

        num_features = self.backbone.num_features

        self.classifier_head = nn.Sequential(
            nn.Linear(num_features, 512),        # FC1: Linear transformation
            nn.BatchNorm1d(512),                 # Batch Normalization
            nn.ReLU(),                           # ReLU activation
            nn.Dropout(p=0.4),                   # Dropout
            nn.Linear(512, num_classes)   # FC2: Final prediction
        )

    def forward(self, x):
        features = self.backbone(x)
        
        out = self.classifier_head(features)
        return out

