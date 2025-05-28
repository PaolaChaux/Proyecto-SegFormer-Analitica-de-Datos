# src/pages/02_Segformer.py
import streamlit as st
st.set_page_config(page_title="2. Segformer", page_icon="🔍")
st.title("2. Arquitectura detallada de SegFormer")
st.markdown("""
## Introducción

SegFormer propone un diseño simple pero eficiente para segmentación semántica basado en Transformers, que combina un encoder jerárquico con un decoder ligero y sin uso de positional embeddings, permitiendo alta precisión y velocidad.
""")
st.image("data/images/003_segformer_img.png", caption="Segmentacion Semantica", width=1000)
st.image("data/images/ARQUITECTURAIMAGEN.png",
            caption="Arquitectura SegFormer (Encoder + Decoder)")
st.markdown("""
## Encoder: Mix Transformer (MiT)

- El encoder es un Transformer jerárquico con múltiples etapas, inspirado en arquitecturas de visión como Swin Transformer pero con simplificaciones.
- Cada etapa procesa la imagen a distintas resoluciones, extrayendo características con diferentes escalas espaciales (1/4, 1/8, 1/16, 1/32 del tamaño original).
- El encoder emplea **mezcla local y global** mediante la auto-atención eficiente con ventanas y reducción progresiva, logrando un buen balance entre precisión y eficiencia.
- MiT-B0, la versión más ligera usada en este modelo, tiene **3.7 millones de parámetros** en el encoder.

## Decoder: MLP ligero y eficiente

- En lugar de decoders complejos o basados en convoluciones, SegFormer utiliza un **decoder simple basado en multilayer perceptrons (MLP)**.
- El decoder toma las características jerárquicas extraídas en diferentes escalas, las proyecta a un espacio común y las combina para producir las predicciones finales de segmentación.
- Esta simplicidad reduce parámetros y costos computacionales, permitiendo inferencia rápida.
- El decoder tiene alrededor de **0.4 millones de parámetros**.

## Ventajas clave

- **Sin positional embeddings:** SegFormer elimina la necesidad de embeddings posicionales explícitos, usando mecanismos de atención y pooling que mantienen la información espacial y permiten inferencia flexible en imágenes de diferentes tamaños.
- **Jerarquía eficiente:** La arquitectura jerárquica captura tanto detalles finos como contexto global.
- **Alta velocidad y precisión:** En resolución 512x512, SegFormer puede alcanzar hasta **47 FPS** en GPU, con resultados de precisión competitivos en datasets estándares como ADE20k y Cityscapes.
- **Diseño modular:** Facilita adaptación a diferentes tamaños de modelo (MiT-B0 a MiT-B5) y a distintas tareas de segmentación.

## Parámetros y desempeño (MiT-B0)

| Componente  | Parámetros (millones) |
|-------------|-----------------------|
| Encoder     | 3.7                   |
| Decoder     | 0.4                   |
| **Total**   | **~4.1**              |

## Referencias

- Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers", *NeurIPS 2021*.
- [Paper completo](https://arxiv.org/abs/2105.15203)
""")