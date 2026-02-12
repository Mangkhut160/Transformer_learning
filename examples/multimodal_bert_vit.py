"""
简易版多模态示例：使用轻量级“BERT”文本编码器 + “ViT”图像编码器，再做特征融合。
本示例仅依赖 numpy，强调概念流程，便于快速理解。
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from transformer_framework import TransformerEncoder


def build_vocab(corpus: List[str]) -> Tuple[dict, dict]:
    """
    根据语料构建一个极简词表。仅用于演示，真实场景请使用成熟 tokenizer。
    """
    words = set()
    for line in corpus:
        for w in line.split():
            words.add(w)
    word2id = {"[PAD]": 0, "[CLS]": 1, "[UNK]": 2}
    for i, w in enumerate(sorted(words), start=3):
        word2id[w] = i
    id2word = {i: w for w, i in word2id.items()}
    return word2id, id2word


def simple_tokenize(text: str, vocab: dict, max_len: int) -> np.ndarray:
    """
    极简 tokenizer：空格切词，前后添加 [CLS]，不足补 PAD。
    """
    tokens = text.split()
    ids = [vocab.get("[CLS]")]
    for t in tokens[: max_len - 1]:
        ids.append(vocab.get(t, vocab["[UNK]"]))
    # padding
    if len(ids) < max_len:
        ids.extend([vocab["[PAD]"]] * (max_len - len(ids)))
    return np.array(ids, dtype=np.int32)


class TextEncoder:
    """
    轻量 BERT 式文本编码器：token embedding + position embedding + TransformerEncoder。
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, ffn_hidden: int, max_len: int):
        self.embed_dim = embed_dim
        scale = 1.0 / math.sqrt(embed_dim)
        self.token_embedding = np.random.randn(vocab_size, embed_dim).astype(np.float32) * scale
        self.position_embedding = np.random.randn(max_len, embed_dim).astype(np.float32) * scale
        self.encoder = TransformerEncoder(num_layers=num_layers, embed_dim=embed_dim, num_heads=num_heads, ffn_hidden=ffn_hidden)

    def __call__(self, token_ids: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # token_ids: (batch, seq)
        batch, seq_len = token_ids.shape
        token_emb = self.token_embedding[token_ids]  # (B, L, D)
        pos_emb = self.position_embedding[:seq_len][None, :, :]  # (1, L, D)
        x = token_emb + pos_emb
        return self.encoder(x, mask=mask)


class PatchEmbedding:
    """
    将图片切成不重叠 patch，然后线性投影到 embed_dim。
    输入假设形状 (B, C, H, W)，H/ W 都能被 patch_size 整除。
    """

    def __init__(self, in_channels: int, patch_size: int, embed_dim: int) -> None:
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        scale = 1.0 / math.sqrt(in_channels * patch_size * patch_size)
        self.proj = np.random.randn(in_channels * patch_size * patch_size, embed_dim).astype(np.float32) * scale

    def __call__(self, x: np.ndarray) -> np.ndarray:
        B, C, H, W = x.shape
        ph = pw = self.patch_size
        assert H % ph == 0 and W % pw == 0, "H 和 W 必须能被 patch_size 整除"
        patches = []
        for i in range(0, H, ph):
            for j in range(0, W, pw):
                patch = x[:, :, i : i + ph, j : j + pw]  # (B, C, ph, pw)
                patch = patch.reshape(B, -1)  # (B, C*ph*pw)
                patches.append(patch)
        patch_seq = np.stack(patches, axis=1)  # (B, num_patches, patch_dim)
        return patch_seq @ self.proj  # 线性投影到 embed_dim


class ViTEncoder:
    """
    极简 ViT：PatchEmbedding + 可选的 cls token + TransformerEncoder。
    """

    def __init__(self, in_channels: int, patch_size: int, embed_dim: int, num_layers: int, num_heads: int, ffn_hidden: int):
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.cls_token = np.zeros((1, 1, embed_dim), dtype=np.float32)
        self.pos_embedding = None  # 按需延迟初始化
        self.encoder = TransformerEncoder(num_layers=num_layers, embed_dim=embed_dim, num_heads=num_heads, ffn_hidden=ffn_hidden)

    def _init_pos(self, num_patches: int, embed_dim: int) -> None:
        scale = 1.0 / math.sqrt(embed_dim)
        self.pos_embedding = np.random.randn(1, num_patches + 1, embed_dim).astype(np.float32) * scale

    def __call__(self, images: np.ndarray) -> np.ndarray:
        # images: (B, C, H, W)
        x = self.patch_embed(images)  # (B, P, D)
        B, P, D = x.shape

        if self.pos_embedding is None:
            self._init_pos(P, D)

        cls = np.repeat(self.cls_token, repeats=B, axis=0)  # (B, 1, D)
        x = np.concatenate([cls, x], axis=1)  # (B, 1+P, D)
        x = x + self.pos_embedding
        return self.encoder(x)


class MultiModalFusion:
    """
    文本 + 图像特征融合：简单拼接后做线性映射。
    """

    def __init__(self, text_dim: int, image_dim: int, fused_dim: int) -> None:
        scale = 1.0 / math.sqrt(text_dim + image_dim)
        self.W = np.random.randn(text_dim + image_dim, fused_dim).astype(np.float32) * scale
        self.b = np.zeros((fused_dim,), dtype=np.float32)

    def __call__(self, text_feat: np.ndarray, image_feat: np.ndarray) -> np.ndarray:
        # text_feat: (B, Lt, D1) -> 取 [CLS] 位置作为聚合
        # image_feat: (B, Li, D2) -> 取 [CLS] 位置
        text_cls = text_feat[:, 0, :]
        image_cls = image_feat[:, 0, :]
        fused = np.concatenate([text_cls, image_cls], axis=-1)
        return np.matmul(fused, self.W) + self.b


def demo_run() -> None:
    """
    演示多模态前向流程：文本经过 TextEncoder，图像经过 ViTEncoder，再融合。
    """
    np.random.seed(123)
    texts = ["a small cat", "a lovely dog"]
    vocab, _ = build_vocab(texts)
    max_len = 6
    token_ids = np.stack([simple_tokenize(text, vocab, max_len) for text in texts], axis=0)

    text_encoder = TextEncoder(
        vocab_size=len(vocab), embed_dim=32, num_layers=1, num_heads=4, ffn_hidden=64, max_len=max_len
    )
    text_feat = text_encoder(token_ids, mask=None)

    # 模拟两张彩色小图像 (B, C, H, W)
    images = np.random.randn(len(texts), 3, 16, 16).astype(np.float32)
    vit_encoder = ViTEncoder(in_channels=3, patch_size=4, embed_dim=32, num_layers=1, num_heads=4, ffn_hidden=64)
    image_feat = vit_encoder(images)

    fusion = MultiModalFusion(text_dim=32, image_dim=32, fused_dim=16)
    fused = fusion(text_feat, image_feat)

    print("文本特征形状:", text_feat.shape)
    print("图像特征形状:", image_feat.shape)
    print("融合后特征形状:", fused.shape)
    print("融合特征示例:\n", fused)


if __name__ == "__main__":
    demo_run()
