from turtle import forward
import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x




class OCET(nn.Module):
    def __init__(self, *, num_classes, dim, depth, heads, dim_head, mlp_dim, dropout=0.,):
        super().__init__()


        self.oce = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(4, 1), padding='same'),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(2),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(4),
            nn.Conv2d(in_channels=4, out_channels=40, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(40),
        )
        

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.fc = nn.Linear(dim, num_classes)
        

    def forward(self, x):
        x = x + self.oce(x)

        x = self.conv1(x)
        x = torch.squeeze(x)

        x = self.transformer(x)
        # get the cls token
        x = x[:, 0]

        x = self.fc(x)
        
        return x


class OCET2D(nn.Module):
    def __init__(self, *, num_classes, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()

        # OCE layer
        self.oce = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(4, 1), padding='same'),
            nn.Sigmoid()
        )

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=dim, kernel_size=(1, 5)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(dim),
        )

        # Transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        device = x.device  # 获取输入张量所在设备

        # Input shape: [batch_size, seq_len, height, width]
        batch_size, seq_len, height, width = x.shape

        # Permute dimensions to [batch_size, width, seq_len, height] for convolution
        x = x.permute(0, 3, 1, 2)  # New shape: [batch_size, width, seq_len, height]

        # Apply OCE layer
        x = x.reshape(batch_size * width, 1, seq_len, height)  # Reshape to [batch_size * width, 1, seq_len, height]
        x = x + self.oce(x)  # Apply OCE
        x = x.reshape(batch_size, width, seq_len, height)  # Reshape back to [batch_size, width, seq_len, height]

        # Apply convolutional layers
        x = self.conv1(x)  # Output shape: [batch_size, dim, seq_len, 1]

        # Squeeze the last dimension and transpose to [batch_size, seq_len, dim]
        x = x.squeeze(-1).transpose(1, 2)  # Output shape: [batch_size, seq_len, dim]

        #### IMPORTANT MODIFICATION ####
        # Transformer expects input of shape [batch_size, seq_len, 100]
        # Here we use a linear layer to project the feature dimension to 100 if it's not already 100
        if x.shape[-1] != 100:
            x = nn.Linear(x.shape[-1], 100).to(device)(x)  # Project the feature dimension to 100 and move to device
        #### END OF MODIFICATION ####

        # Pass through the Transformer
        x = self.transformer(x)  # Transformer expects input of shape [batch_size, seq_len, 100]

        # Apply global average pooling along the sequence length dimension
        x = x.mean(dim=1)  # Shape after pooling: [batch_size, 100]

        # Final fully connected layer for classification
        x = self.fc(x)

        return x


class ImprovedOCET(nn.Module):
    def __init__(self, *, num_classes, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()

        self.oce = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(4, 1), padding='same'),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.attention_pooling = nn.MultiheadAttention(embed_dim=dim, num_heads=1, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes)
        )

    def forward(self, x):
        x = x + self.oce(x)  # Shape remains (batch_size, 1, height, width)

        x = self.conv1(x)  # Expected output: (batch_size, 64, height, new_width)

        x = x.flatten(2)  # Flatten the height dimension into width, reshape to (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, sequence_length, channels)

        x = self.transformer(x)  # Input should be (batch_size, sequence_length, dim)

        # Attention pooling
        query = x.mean(dim=1, keepdim=True)
        x, _ = self.attention_pooling(query, x, x)
        x = x.squeeze(1)

        x = self.fc(x)  # Fully connected layer

        return x

