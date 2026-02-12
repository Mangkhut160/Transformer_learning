# Transformer_learning

## 内容简介
提供最小可运行的 Transformer 教学示例，以及对应的挖空练习代码，帮助快速理解自注意力、BERT 与 ViT 的多模态融合流程。所有示例仅依赖 numpy，可通过 `pip install numpy` 安装。

## 文件说明
- `examples/transformer_framework.py`：完整的 Transformer 编码器示例，含详细中文注释与演示入口。
- `examples/transformer_framework_exercise.py`：Transformer 挖空练习版，填补 `____` 处即可运行。
- `examples/multimodal_bert_vit.py`：轻量 BERT（文本）+ ViT（图像）多模态融合示例，含中文讲解。
- `examples/multimodal_bert_vit_exercise.py`：多模态挖空练习版。

## 运行方式
```bash
cd examples
python transformer_framework.py
python multimodal_bert_vit.py
```

若进行练习，请先补全对应的 `____`，再运行相同命令验证输出形状。
