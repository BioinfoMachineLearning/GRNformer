
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class TransformerAutoencoder(nn.Module):
    def __init__(self, embed_dim=128, nhead=8, num_layers=6):
        super(TransformerAutoencoder, self).__init__()

        # Encoder: Conv2d to project input feature dimension n to embed_dim, keeping the shape [1, m, n] unchanged
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=1, padding=0)

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,batch_first=True),
            num_layers=num_layers
        )

        # Decoder: Conv2d transpose to project embed_dim back to original feature dimension n
        self.deconv2d = nn.ConvTranspose1d(in_channels=embed_dim, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        # Input shape: [batch_size, m, n]
        batch_size, seq_len, feature_dim = x.shape
        
        # Reshape input to add a channel dimension for Conv2d: [batch_size, 1, m, n]
        x = x.permute(1,0,2).float()

        # 1. Encoder
        x = self.conv1d(x)  # Apply Conv2d, project feature dimension n to embed_dim, keep shape [batch_size, embed_dim, m, n]
        #print(x.shape)
        #x = x.squeeze(1)  # Remove the channel dimension, now shape is [batch_size, embed_dim, m]
        
        # Permute to [m, batch_size, embed_dim] for Transformer
        x = x.permute(2, 0, 1)
        latent = self.transformer_encoder(x)  # Transformer encoder
        latent = latent.permute(1, 2, 0)  # Shape back to [batch_size, embed_dim, m]
        #print(latent.shape)
        latent1 =  torch.mean(latent, dim=-1, keepdim=True) 
        #print(latent1.shape)
        # 2. Decoder
        #decoded = latent  # Add channel dimension back
        #decoded = self.deconv2d(decoded)  # Apply Transposed Conv2d, project embed_dim back to original feature dimension n
        #decoded =  torch.mean(decoded, dim=1, keepdim=True)
        #decoded = decoded.permute(1,0,2)
           # Shape: [batch_size, m, n]
        #print(decoded.shape)
        return  latent1.permute(2, 0, 1)  # Return reconstructed output and latent space




    '''def __init__(self, embed_dim=128, num_heads=4, num_layers=2, hidden_dim=256):
        super(TransformerAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=1, stride=1)  # [1, m, n] -> [embed_dim, m, n]
        )
        
        # Transformer Encoder
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=num_layers
        )

        # Linear layer to reduce features from (n * embed_dim) to embed_dim
        self.linear = nn.Linear(embed_dim, embed_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 1, kernel_size=1, stride=1),  # [embed_dim, m, n] -> [1, m, n]
            nn.Sigmoid()  # Scale output to [0, 1]
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        # x shape: [batch_size, m, n]
        batch_size, m, n = x.size()
        x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, m, n]
        
        # Encode
        x = self.encoder(x)  # Output shape: [batch_size, embed_dim, m, n]
        
        # Prepare for Transformer
        # Reshape to [sequence_length, batch_size, embed_dim]
        x = x.permute(0, 2, 3, 1).reshape(batch_size, m, n, -1)  # Shape: [batch_size, m, n, embed_dim]
        x = x.permute(0, 2, 1, 3).reshape(batch_size, n, m, -1)  # Shape: [batch_size, n, m, embed_dim]
        x = x.permute(2, 0, 1, 3).reshape(m, batch_size, n * self.embed_dim)  # Shape: [m, batch_size, n * embed_dim]
        
        # Transformer Encoding
        x = self.transformer_encoder(x)  # Output shape: [m, batch_size, n * embed_dim]
        
        # Apply Linear layer to reduce feature dimension
        x = x.permute(1, 2, 0)  # Shape: [batch_size, n * embed_dim, m]
        x = self.linear(x)  # Shape: [batch_size, embed_dim, m]
        x = x.permute(0, 2, 1)  # Shape: [batch_size, m, embed_dim]
        
        # Reshape Transformer output
        x = x.reshape(batch_size, self.embed_dim, m, n).permute(0, 2, 1, 3)  # Shape: [batch_size, embed_dim, m, n]
        
        # Decode
        x = self.decoder(x)  # Output shape: [batch_size, 1, m, n]
        
        x = x.squeeze(1)  # Remove channel dimension: [batch_size, m, n]
        return x
'''

'''
class ConvAutoencoder(nn.Module):
    def __init__(self, embed_dim=128):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [1, m, n] -> [16, m/2, n/2]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [16, m/2, n/2] -> [32, m/4, n/4]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [32, m/4, n/4] -> [64, m/8, n/8]
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1), # [64, m/8, n/8] -> [128, m/16, n/16]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # [128, m/16, n/16] -> [128, 1, 1]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=4, stride=2, padding=1), # [128, 1, 1] -> [64, 2, 2]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # [64, 2, 2] -> [32, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # [32, 4, 4] -> [16, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1), # [16, 8, 8] -> [1, 16, 16]
            #nn.Sigmoid() # Scale output to [0, 1]
        )

    def forward(self, x):
        # x shape: [batch_size, m, n]
        x = x.unsqueeze(1).float()  # Add channel dimension: [batch_size, 1, m, n]
        
        # Encode
        x = self.encoder(x)  # Output shape: [batch_size, 128, 1, 1]
        Latent_embed = x
        # Decode
        x = self.decoder(x)  # Output shape: [batch_size, 1, m, n]
        
        x = x.squeeze(1)  # Remove channel dimension: [batch_size, m, n]
        print(x.shape)
        return x, Latent_embed

        '''