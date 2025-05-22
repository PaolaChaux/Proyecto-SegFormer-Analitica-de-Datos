
# Arquitectura de SegFormer

![Arquitectura de SegFormer](https://github.com/PaolaChaux/Proyecto-SegFormer-Analitica-de-Datos/blob/main/data/images/ARQUITECTURAIMAGEN.png)

SegFormer es un modelo de segmentación semántica que combina la eficiencia de los Transformers con un decodificador ligero basado en MLP. Su arquitectura se centra en dos bloques principales: un codificador jerárquico (MiT) y un decodificador All-MLP.

---
# SegFormer Architecture

SegFormer is a semantic segmentation model that combines the efficiency of Transformers with a lightweight MLP-based decoder. Its architecture is centered around two main blocks: a hierarchical Transformer encoder (MiT) and an All-MLP decoder.

---

## 1. Encoder (MiT: Mix Transformer)

The encoder in SegFormer serves as the backbone of the model. It is designed as a hierarchical Transformer that produces multi-scale feature representations, which is essential for semantic segmentation.

### Key Features:

- **Hierarchical levels**: Generates feature maps at multiple resolutions: `H/4`, `H/8`, `H/16`, `H/32`.
- **Overlapping Patch Embedding**: Unlike ViT, the patches are overlapping and processed via convolutions to preserve local continuity.
- **No Positional Encoding**: Instead of fixed positional embeddings, a 3x3 convolution inside each FFN captures relative spatial information.
- **Efficient Self-Attention**: Reduces attention complexity from $O(N^2)$ to $O(N^2/R)$ using sequence reduction.

### Encoder Flow:

1. **Input image**: Size `H x W x 3`
2. **Divided into overlapping patches** (e.g., `4x4`)
3. **Passed through 4 Transformer stages**
4. **Produces 4 feature maps** at different resolutions (`S1` to `S4`)

---

## 2. Decoder (All-MLP)

The decoder in SegFormer is extremely simple. It avoids heavy designs like ASPP or convolutional layers and uses only MLPs.

### Decoder Flow:

1. Each of the 4 encoder outputs passes through an **MLP** to unify the channel dimensions.
2. All feature maps are **upsampled** to the same resolution: $H/4 \times W/4$.
3. They are **concatenated** into a single tensor.
4. A final **MLP** predicts a tensor of shape:

$$
\frac{H}{4} \times \frac{W}{4} \times N_{cls}
$$

Where $N_{cls}$ is the number of segmentation classes.

5. `argmax` is applied along the class dimension to obtain the segmentation mask:

$$
\frac{H}{4} \times \frac{W}{4}
$$

This mask can be upsampled back to the original image resolution $H \times W$ for visualization or post-processing.

---

## Model Input

The model expects RGB images (3 channels) that are converted into tensors using the `SegformerImageProcessor`.

```python
processor(images=image, return_tensors="pt")
```
## Model Output

- **Logits**: A tensor of shape `B x N_cls x H/4 x W/4` (where B is the batch size).
- **Segmentation mask**: After `argmax`, a mask of shape `H/4 x W/4` is obtained.
- **Inference time**: Highly optimized. For example, SegFormer-B0 can reach up to 48 FPS.

---

## Advantages of the SegFormer Architecture

- **No Positional Embedding**: Avoids the need to interpolate positional encodings when input size changes.
- **More efficient and lightweight** than architectures like SETR and DeepLabv3+.
- **Stronger robustness** to common corruptions (e.g., noise, weather, JPEG artifacts).
- **Hierarchical Transformer Backbone (MiT)** captures both local details and global context.
- **Simplified decoder** with only MLPs reduces computational cost.
- **Better speed-accuracy trade-off** than other Transformer-based models.

---

## Reference

Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers**. _arXiv preprint arXiv:2105.15203_.

