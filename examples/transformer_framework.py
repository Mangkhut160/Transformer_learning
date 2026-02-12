"""
一个最小可运行的 Transformer 编码器示例，包含详细中文注释，便于理解核心组件。

特点：
1. 仅依赖 numpy，方便在任意环境下直接运行学习。
2. 代码使用模块化的方式展示多头自注意力、前馈网络、层归一化以及残差连接。
3. 在 __main__ 中给出最小示例，帮助快速验证计算流程。
"""

import math
from typing import Optional

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    简单稳定的 softmax 实现，先减去最大值避免指数溢出。
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class LayerNorm:
    """
    层归一化：对最后一个维度做均值方差标准化，再乘以可学习的 gamma 和 beta。
    这里的 gamma、beta 采用简单的可训练向量初始化为 1 和 0。
    """

    def __init__(self, feature_dim: int, eps: float = 1e-5) -> None:
        self.eps = eps
        self.gamma = np.ones((feature_dim,), dtype=np.float32)
        self.beta = np.zeros((feature_dim,), dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * norm + self.beta


class MultiHeadSelfAttention:
    """
    多头自注意力层（仅实现最小示例）。
    步骤：
    1. 输入 x 映射为 q、k、v。
    2. 拆分为多个头，计算缩放点积注意力。
    3. 连接所有头，映射回输出维度。
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        assert (
            embed_dim % num_heads == 0
        ), "embed_dim 必须能被 num_heads 整除，便于拆分头部"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 权重矩阵初始化（使用较小的随机数避免数值过大）
        scale = 1.0 / math.sqrt(embed_dim)
        self.W_q = np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale
        self.W_k = np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale
        self.W_v = np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale
        self.W_o = np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        将 (batch, seq, embed_dim) 拆成 (batch, num_heads, seq, head_dim)。
        """
        bsz, seq_len, _ = x.shape
        x = x.reshape(bsz, seq_len, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        将 (batch, num_heads, seq, head_dim) 还原成 (batch, seq, embed_dim)。
        """
        bsz, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(bsz, seq_len, num_heads * head_dim)

    def __call__(
        self, x: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # 1) 线性映射得到 q, k, v
        q = np.matmul(x, self.W_q)
        k = np.matmul(x, self.W_k)
        v = np.matmul(x, self.W_v)

        # 2) 拆分头
        q = self.split_heads(q)  # (B, H, L, D)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # 3) 点积注意力
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        if mask is not None:
            # 将 mask 为 0 的位置设为极小值，避免被 softmax 选中
            scores = scores + (mask[:, None, None, :] * -1e9)

        attn = softmax(scores, axis=-1)
        context = np.matmul(attn, v)

        # 4) 连接各个头
        context = self.combine_heads(context)

        # 5) 输出线性层
        out = np.matmul(context, self.W_o)
        return out


class PositionwiseFFN:
    """
    前馈网络：两个线性层，中间用 ReLU 激活。
    """

    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        scale = 1.0 / math.sqrt(embed_dim)
        self.W1 = np.random.randn(embed_dim, hidden_dim).astype(np.float32) * scale
        self.b1 = np.zeros((hidden_dim,), dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, embed_dim).astype(np.float32) * scale
        self.b2 = np.zeros((embed_dim,), dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.matmul(x, self.W1) + self.b1
        x = np.maximum(0, x)  # ReLU
        x = np.matmul(x, self.W2) + self.b2
        return x


class TransformerEncoderLayer:
    """
    单层 Transformer 编码器：自注意力 + 残差 + LayerNorm + 前馈 + 残差 + LayerNorm
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden: int) -> None:
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = PositionwiseFFN(embed_dim, ffn_hidden)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # 自注意力 + 残差
        attn_out = self.attn(x, mask)
        x = self.norm1(x + attn_out)

        # 前馈 + 残余
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class TransformerEncoder:
    """
    多层堆叠的 Transformer 编码器。
    """

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
    使用随机输入演示一遍前向过程，方便快速验证逻辑是否跑通。
    """
    np.random.seed(42)
    batch, seq_len, embed_dim = 2, 5, 16
    num_heads = 4
    ffn_hidden = 32

    # 构造随机输入
    sample_input = np.random.randn(batch, seq_len, embed_dim).astype(np.float32)
    # 可选 mask：这里模拟 padding，后两个 token 无效
    mask = np.array(
        [
            [0, 0, 0, -1e9, -1e9],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    encoder = TransformerEncoder(num_layers=2, embed_dim=embed_dim, num_heads=num_heads, ffn_hidden=ffn_hidden)
    output = encoder(sample_input, mask=mask)

    print("输入形状:", sample_input.shape)
    print("输出形状:", output.shape)
    print("输出示例（前两个时间步）:\n", output[:, :2, :4])  # 仅打印部分便于阅读


if __name__ == "__main__":
    demo_run()
