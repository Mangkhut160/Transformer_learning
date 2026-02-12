"""
多模态 BERT + ViT 挖空练习。
请对照 multimodal_bert_vit.py 补全 ____，理解文本、图像编码以及融合的流程。
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from transformer_framework import TransformerEncoder


def build_vocab(corpus: List[str]) -> Tuple[dict, dict]:
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
    tokens = text.split()
    ids = [vocab.get("[CLS]")]
    for t in tokens[: max_len - 1]:
        ids.append(vocab.get(t, vocab["[UNK]"]))
    if len(ids) < max_len:
        ids.extend([vocab["[PAD]"]] * (max_len - len(ids)))
    return np.array(ids, dtype=np.int32)


class TextEncoder:
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, ffn_hidden: int, max_len: int):
        self.embed_dim = embed_dim
        scale = 1.0 / math.sqrt(embed_dim)
        self.token_embedding = np.random.randn(vocab_size, embed_dim).astype(np.float32) * scale
        self.position_embedding = np.random.randn(max_len, embed_dim).astype(np.float32) * scale
        self.encoder = TransformerEncoder(num_layers=num_layers, embed_dim=embed_dim, num_heads=num_heads, ffn_hidden=ffn_hidden)

    def __call__(self, token_ids: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        batch, seq_len = token_ids.shape
        token_emb = ____  # 提示：根据 token_ids 查 embedding
        pos_emb = ____  # 提示：取前 seq_len 个位置编码并 broadcast
        x = token_emb + pos_emb
        return ____  # 提示：送入 TransformerEncoder


class PatchEmbedding:
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int) -> None:
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        scale = 1.0 / math.sqrt(in_channels * patch_size * patch_size)
        self.proj = np.random.randn(in_channels * patch_size * patch_size, embed_dim).astype(np.float32) * scale

    def __call__(self, x: np.ndarray) -> np.ndarray:
        B, C, H, W = x.shape
        ph = pw = self.patch_size
        assert H % ph == 0 and W % pw == 0
        patches = []
        for i in range(0, H, ph):
            for j in range(0, W, pw):
                patch = x[:, :, i : i + ph, j : j + pw]
                patch = patch.reshape(B, -1)
                patches.append(patch)
        patch_seq = np.stack(patches, axis=1)
        return patch_seq @ ____  # 提示：线性投影矩阵


class ViTEncoder:
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int, num_layers: int, num_heads: int, ffn_hidden: int):
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.cls_token = np.zeros((1, 1, embed_dim), dtype=np.float32)
        self.pos_embedding = None
        self.encoder = TransformerEncoder(num_layers=num_layers, embed_dim=embed_dim, num_heads=num_heads, ffn_hidden=ffn_hidden)

    def _init_pos(self, num_patches: int, embed_dim: int) -> None:
        scale = 1.0 / math.sqrt(embed_dim)
        self.pos_embedding = np.random.randn(1, num_patches + 1, embed_dim).astype(np.float32) * scale

    def __call__(self, images: np.ndarray) -> np.ndarray:
        x = self.patch_embed(images)
        B, P, D = x.shape
        if self.pos_embedding is None:
            self._init_pos(P, D)
        cls = np.repeat(self.cls_token, repeats=B, axis=0)
        x = np.concatenate([cls, x], axis=1)
        x = x + ____  # 提示：加上位置编码
        return ____  # 提示：送入 TransformerEncoder


class MultiModalFusion:
    def __init__(self, text_dim: int, image_dim: int, fused_dim: int) -> None:
        scale = 1.0 / math.sqrt(text_dim + image_dim)
        self.W = np.random.randn(text_dim + image_dim, fused_dim).astype(np.float32) * scale
        self.b = np.zeros((fused_dim,), dtype=np.float32)

    def __call__(self, text_feat: np.ndarray, image_feat: np.ndarray) -> np.ndarray:
        text_cls = text_feat[:, 0, :]
        image_cls = image_feat[:, 0, :]
        fused = np.concatenate([text_cls, image_cls], axis=-1)
        return fused @ ____ + ____  # 提示：线性映射


def demo_run() -> None:
    np.random.seed(7)
    texts = ["hello world", "tiny multimodal"]
    vocab, _ = build_vocab(texts)
    max_len = 6
    token_ids = np.stack([simple_tokenize(text, vocab, max_len) for text in texts], axis=0)

    text_encoder = TextEncoder(
        vocab_size=len(vocab), embed_dim=16, num_layers=1, num_heads=4, ffn_hidden=32, max_len=max_len
    )
    text_feat = text_encoder(token_ids, mask=None)

    images = np.random.randn(len(texts), 3, 16, 16).astype(np.float32)
    vit_encoder = ViTEncoder(in_channels=3, patch_size=4, embed_dim=16, num_layers=1, num_heads=4, ffn_hidden=32)
    image_feat = vit_encoder(images)

    fusion = MultiModalFusion(text_dim=16, image_dim=16, fused_dim=8)
    fused = fusion(text_feat, image_feat)
    print("融合后特征:", fused)


if __name__ == "__main__":
    demo_run()
