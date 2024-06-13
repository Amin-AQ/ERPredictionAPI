import torchvision.models as models
import torch
import os
import torch.nn as nn
class Resnet50(nn.Module):
    def __init__(self,num_classes):
        #define necessary layers
        super().__init__()
        self.num_classes = num_classes
        self.model = models.resnet50(weights=True)
        
        # Unfreeze model weights
        for param in self.model.parameters():
            param.requires_grad = False 
        
    def forward(self,X):
        #define forward pass here
        X = self.model(X)
        return X    

def get_feature_extractor():
    patch_feature_extractor = Resnet50(1) 
    state_dict = torch.load('./models/resnet50-specialized.pt', map_location=torch.device('cpu'))
    patch_feature_extractor.load_state_dict(state_dict)
    patch_feature_extractor.eval()  
    return patch_feature_extractor