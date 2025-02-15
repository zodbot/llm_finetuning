import torch
import torch.nn as nn

from src.attention import MultiHeadAttention
from src.feed_forward import FeedForward
from src.layers import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # print("transformer block: ")
        # print("input size: ", x.shape)
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        # print("after multi-head", x.shape)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        # print("after ff", x.shape)
        x = self.drop_shortcut(x)
        x = x + shortcut
        # print("final: ", x.shape)
        return x
