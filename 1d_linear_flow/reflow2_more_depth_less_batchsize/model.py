import torch
import math
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# -------------------------
# Model Definition
# -------------------------
# A small network that takes x_t and t and predicts e
class SinusoidalTimeEmbedding(nn.Module):
    """
    Creates sinusoidal embeddings for a scalar time t.
    t_emb_dim is the dimension of the embedding.
    Embedding: for t in [0,1], we use frequencies like a positional encoding.
    """
    def __init__(self, t_emb_dim=64, max_period=10000):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.max_period = max_period
        # We don't use parameters here; this is a fixed embedding like positional encoding.

    def forward(self, t):
        # t shape: [B, 1]
        # We'll assume t in [0,1], and map it into angles
        half_dim = self.t_emb_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        # freqs shape: [half_dim]

        # t: [B,1], freqs broadcast to [B, half_dim]
        args = t * freqs.unsqueeze(0)
        # sinusoidal embedding: [sin(2pi args), cos(2pi args)]
        emb = torch.cat([torch.sin(2*math.pi * args), torch.cos(2*math.pi * args)], dim=-1)
        # emb: [B, t_emb_dim]
        return emb

class FiLMConditionedLinear(nn.Module):
    """
    A linear layer that is conditioned on t_emb.
    We use t_emb to produce scale (gamma) and shift (beta) parameters
    that modulate the hidden features after a linear transform.

    FiLM: h = gamma * (Wx + b) + beta
    """
    def __init__(self, in_dim, out_dim, t_emb_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        # Maps time embedding to scale and shift
        self.film = nn.Linear(t_emb_dim, out_dim*2)

    def forward(self, x, t_emb):
        # x shape: [B, in_dim]
        # t_emb shape: [B, t_emb_dim]
        out = self.linear(x)
        gamma_beta = self.film(t_emb)  # [B, 2*out_dim]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        # Apply FiLM
        out = gamma * out + beta
        return out

class ResidualFiLMBlock(nn.Module):
    """
    A residual MLP block that uses FiLM conditioning at each layer.
    """
    def __init__(self, dim, t_emb_dim, hidden_dim=128):
        super().__init__()
        self.lin1 = FiLMConditionedLinear(dim, hidden_dim, t_emb_dim)
        self.act = nn.SiLU()
        self.lin2 = FiLMConditionedLinear(hidden_dim, dim, t_emb_dim)
        self.res_scale = nn.Parameter(torch.ones(1))  # optional: learnable residual scale

    def forward(self, x, t_emb):
        # x: [B, dim]
        h = self.lin1(x, t_emb)
        h = self.act(h)
        h = self.lin2(h, t_emb)
        return x + self.res_scale * h

class SimpleNet(nn.Module):
    """
    Improved model architecture:
    - Embed t into t_emb using sinusoidal embedding.
    - Multiple residual FiLM blocks for feature processing.
    - Predict v from (x_t, t).

    Assume input: x_t: [B, x_dim], t: [B, 1]
    Output: v: [B, x_dim]
    """
    def __init__(self, x_dim=1, t_emb_dim=64, hidden_dim=128, num_blocks=4):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(t_emb_dim=t_emb_dim)
        # Initial linear to bring x into a hidden dimension
        self.input_proj = nn.Linear(x_dim, hidden_dim)

        # A series of residual FiLM blocks
        self.blocks = nn.ModuleList([
            ResidualFiLMBlock(hidden_dim, t_emb_dim, hidden_dim=hidden_dim)
            for _ in range(num_blocks)
        ])

        # Output layer to predict v
        self.output_proj = nn.Linear(hidden_dim, x_dim)

    def forward(self, x, t):
        # x: [B, x_dim]
        # t: [B, 1]
        t_emb = self.time_embedding(t)  # [B, t_emb_dim]

        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)

        v = self.output_proj(h)
        return v

#class SimpleNet(nn.Module):
#    def __init__(self, input_dim=2, hidden_dim=64):
#        super(SimpleNet, self).__init__()
#        self.net = nn.Sequential(
#            nn.Linear(input_dim, hidden_dim),
#            nn.ReLU(),
#            nn.Linear(hidden_dim, hidden_dim),
#            nn.ReLU(),
#            nn.Linear(hidden_dim, 1)
#        )
#
#    def forward(self, xt, t):
#        # input is concatenation of xt and t
#        inp = torch.cat([xt, t], dim=-1)
#        return self.net(inp)

class SimpleNet2(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_embed_dim):
        super(SimpleNet2, self).__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU()
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # Output same size as x_t
        )

    def forward(self, x_t, t):
        # Time embedding
        t = t.view(-1, 1)  # Ensure t is 2D (batch_size, 1)
        t_embed = self.time_embed(t)

        # Concatenate x_t with time embedding
        x_t = torch.cat([x_t, t_embed], dim=-1)

        # Pass through fully connected layers
        return self.fc(x_t)
