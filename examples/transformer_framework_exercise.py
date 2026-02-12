"""
Transformer 编码器挖空练习版本。
请根据注释补全 ____ 处的代码，使其功能与 transformer_framework.py 保持一致。
"""

import math
from typing import Optional

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class LayerNorm:
    def __init__(self, feature_dim: int, eps: float = 1e-5) -> None:
        self.eps = eps
        self.gamma = np.ones((feature_dim,), dtype=np.float32)
        self.beta = np.zeros((feature_dim,), dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mean = ____  # 提示：对最后一维求均值
        var = ____  # 提示：对最后一维求方差
        norm = (x - mean) / np.sqrt(var + self.eps)
        return ____ * norm + ____  # 提示：分别乘 gamma，加 beta


class MultiHeadSelfAttention:
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        scale = 1.0 / math.sqrt(embed_dim)
        self.W_q = np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale
        self.W_k = np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale
        self.W_v = np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale
        self.W_o = np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        bsz, seq_len, _ = x.shape
        x = x.reshape(bsz, seq_len, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        bsz, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(bsz, seq_len, num_heads * head_dim)

    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # 1) 线性映射 q, k, v
        q = np.matmul(x, ____)
        k = np.matmul(x, ____)
        v = np.matmul(x, ____)

        # 2) 拆分头
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # 3) 缩放点积注意力
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(____)
        if mask is not None:
            scores = scores + (mask[:, None, None, :] * -1e9)
        attn = softmax(scores, axis=-1)
        context = np.matmul(attn, v)

        # 4) 连接头并线性映射
        context = self.combine_heads(context)
        out = np.matmul(context, ____)
        return out


class PositionwiseFFN:
    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        scale = 1.0 / math.sqrt(embed_dim)
        self.W1 = np.random.randn(embed_dim, hidden_dim).astype(np.float32) * scale
        self.b1 = np.zeros((hidden_dim,), dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, embed_dim).astype(np.float32) * scale
        self.b2 = np.zeros((embed_dim,), dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.matmul(x, self.W1) + self.b1
        x = ____  # 提示：使用 ReLU
        x = np.matmul(x, self.W2) + ____
        return x


class TransformerEncoderLayer:
    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden: int) -> None:
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = PositionwiseFFN(embed_dim, ffn_hidden)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + ____)  # 残差 + LayerNorm

        ffn_out = self.ffn(x)
        x = self.norm2(x + ____)
        return x


class TransformerEncoder:
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int, ffn_hidden: int):
        self.layers = [
            TransformerEncoderLayer(embed_dim, num_heads, ffn_hidden)
            for _ in range(num_layers)
        ]

    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        for layer in self.layers:
            x = layer(x, mask)
        return x


def demo_run() -> None:
    """
    运行前请先完成上面的填空。
    """
    np.random.seed(0)
    batch, seq_len, embed_dim = 2, 4, 8
    encoder = TransformerEncoder(num_layers=1, embed_dim=embed_dim, num_heads=2, ffn_hidden=16)
    sample_input = np.random.randn(batch, seq_len, embed_dim).astype(np.float32)
    out = encoder(sample_input, mask=None)
    print("输出形状:", out.shape)


if __name__ == "__main__":
    demo_run()
