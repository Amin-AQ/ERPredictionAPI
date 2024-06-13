import math 
import torchvision.models as models
import torch
import torch.nn as nn
from torch.nn import functional as F

class BagPipeline(nn.Module):
    def __init__(self,num_classes):
        #define necessary layers
        super().__init__()
        self.num_classes = num_classes
        
        # Load pre-trained weights
        self.base = models.efficientnet_b0(weights=True)
        self.base.features[0][0] = nn.Conv2d(1, 32,kernel_size= (3,3), stride = 2, padding= 1, bias=False)
        # freeze model weights
        for param in self.base.parameters():
            param.requires_grad = False 
        
        self.flatten = nn.Flatten()
        
        self.head = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ) 
    
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
    
    def forward(self,X):
        attended_values, attention_scores = self.attention(X, X, X)
        X = self.base(attended_values)
        X = self.flatten(X)
        X = self.head(X)
        return X, F.sigmoid(X)

def get_bag_model():
    bag_model = BagPipeline(1)
    bag_model.load_state_dict(torch.load('./models/attention-based-bag-level-model.pt',map_location=torch.device('cpu')))
    bag_model.eval()
    return bag_model


