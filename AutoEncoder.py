from sklearn.preprocessing import StandardScaler 
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
           nn.Linear(input_size, 16),
           nn.ReLU(),
           nn.Linear(16, encoding_dim),
           nn.ReLU()
        )
        self.decoder = nn.Sequential(
           nn.Linear(encoding_dim, 16),
           nn.ReLU(),
           nn.Linear(16, input_size),
           nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_reduced_patch_embeddings(embeddings):
    
    input_size = 1000  # Number of input features
    encoding_dim = 224  # Desired number of output dimensions
    auto_encoder = Autoencoder(input_size, encoding_dim)
    
    # Loading pretrained autoencoder model
    auto_encoder.load_state_dict(torch.load('./models/autoencoder.pt',map_location=torch.device('cpu')))
    auto_encoder.eval()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    X_tensor = torch.FloatTensor(X_scaled)

    # Encoding the data using the trained autoencoder
    reduced_embeddings = auto_encoder.encoder(X_tensor).detach().numpy()
    
    return reduced_embeddings 
