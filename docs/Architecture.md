
# Arquitectura de SegFormer

![Arquitectura de SegFormer](https://github.com/PaolaChaux/Proyecto-SegFormer-Analitica-de-Datos/blob/main/data/images/ARQUITECTURAIMAGEN.png)

# Arquitectura de SegFormer

SegFormer es un modelo de segmentación semántica que combina la eficiencia de los Transformers con un decodificador ligero basado en MLP. Su arquitectura se centra en dos bloques principales: un codificador jerárquico (MiT) y un decodificador All-MLP.

---

## 1. Encoder (Codificador - MiT: Mix Transformer)

El encoder de SegFormer actúa como el backbone del modelo. Está diseñado como un Transformer jerárquico que produce representaciones multi-escala, lo cual es esencial para la segmentación semántica.

### Características clave:

- **Jerarquía de niveles**: genera características en múltiples resoluciones: `H/4`, `H/8`, `H/16`, `H/32`.
- **Patch Embedding con solapamiento**: a diferencia de ViT, los parches no son independientes. Se usa convolución con solapamiento para preservar continuidad espacial.
- **Sin Positional Encoding**: en lugar de embeddings posicionales, utiliza una convolución 3x3 dentro de cada FFN para capturar ubicación relativa.
- **Efficient Self-Attention**: reduce la secuencia con una tasa de compresión para disminuir la complejidad de atención de $O(N^2)$ a $O(N^2/R)$.

### Flujo:

1. **Imagen de entrada**: tamaño `H x W x 3`
2. **Se divide en parches solapados** (por ejemplo, `4x4`)
3. **Pasan por 4 etapas de bloques Transformer**
4. **Se generan 4 mapas de características** en diferentes resoluciones (`S1` a `S4`)

---

## 2. Decoder (Decodificador All-MLP)

El decodificador de SegFormer es extremadamente simple. No usa convoluciones ni estructuras pesadas como ASPP. Solo MLPs.

### Flujo del decoder:

1. Cada una de las 4 salidas del encoder pasa por un **MLP** para unificar dimensiones de canal.
2. Se **upsamplean** todas a la misma resolución $H/4 \times W/4$.
3. Se **concatenan** las características.
4. Una **MLP final** predice un tensor de tamaño:

$$
\frac{H}{4} \times \frac{W}{4} \times N_{cls}
$$

Donde $N_{cls}$ es el número de clases posibles.

5. Se aplica `argmax` sobre la dimensión de clases para obtener la máscara final:

$$
\frac{H}{4} \times \frac{W}{4}
$$

Esta máscara puede luego interpolarse a $H \times W$ para que coincida con el tamaño original de la imagen.

---

## Entradas del Modelo

El modelo espera imágenes RGB (tres canales) que se convierten en tensores mediante el procesador `SegformerImageProcessor`.

```python
processor(images=image, return_tensors="pt")


## Salidas del Modelo

- **Logits**: tensor de tamaño `B x N_cls x H/4 x W/4` (donde B es el batch size).
- **Máscara de segmentación**: después de `argmax`, se obtiene una máscara `H/4 x W/4`.
- **Tiempo de inferencia**: muy competitivo, por ejemplo, SegFormer-B0 alcanza hasta 48 FPS.

---

## Ventajas de la Arquitectura de SegFormer

- **No usa Positional Embedding**: evita el problema de interpolación cuando la resolución cambia.
- **Más eficiente y ligero** que arquitecturas como SETR y DeepLabv3+.
- **Mayor robustez** ante perturbaciones (ej. ruido, clima, compresión JPEG).
- **Backbone jerárquico (MiT)** captura detalles locales y contexto global.
- **Decoder simplificado** con MLPs que reduce drásticamente el costo computacional.
- **Mejor relación velocidad-precisión** que otros modelos basados en Transformers.

---

## Referencia

Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers**. _arXiv preprint arXiv:2105.15203_.
