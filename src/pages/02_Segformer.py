# src/pages/02_Segformer.py
import streamlit as st

st.set_page_config(
    page_title="2. SegFormer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Esta app fue creada por Marce para demo de Segmentación Semántica."
    }
)

st.markdown("""
<style>
    .main {
        padding: 2rem 4rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: #f5f7fa;
        color: #222222;
    }
    h1, h2, h3 {
        color: #0d6efd;
    }
    table {
        border-collapse: collapse;
        width: 50%;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px 12px;
        text-align: left;
    }
    th {
        background-color: #0d6efd;
        color: white;
    }
    a {
        color: #0d6efd;
        font-weight: bold;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

st.title("2. Arquitectura detallada de SegFormer")

st.markdown("""
## Introducción

SegFormer propone un diseño simple pero eficiente para segmentación semántica basado en Transformers, que combina un encoder jerárquico con un decoder ligero y sin uso de positional embeddings, permitiendo alta precisión y velocidad.
""")

st.image("data/images/003_segformer_img.png", caption="Segmentación Semántica", width=900)
st.image("data/images/000_Architecture.png", caption="Arquitectura SegFormer (Encoder + Decoder)", width=900)

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

<table>
    <tr><th>Componente</th><th>Parámetros (millones)</th></tr>
    <tr><td>Encoder</td><td>3.7</td></tr>
    <tr><td>Decoder</td><td>0.4</td></tr>
    <tr><th>Total</th><th>~4.1</th></tr>
</table>

## Referencias

- Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers", *NeurIPS 2021*.
- [Paper completo](https://arxiv.org/abs/2105.15203)
""", unsafe_allow_html=True)
