import streamlit as st
import os

st.set_page_config(
    page_title="SegFormer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Esta app fue creada por Marce para demo de Segmentación Semántica."
    }
)


st.title("SegFormer")

st.markdown("""
Es un modelo de NVIDIA con una arquitectura reciente basada en Transformers diseñada específicamente para segmentación semántica con eficiencia y precisión.  
SegFormer combina un encoder jerárquico eficiente que captura características a múltiples escalas y un decoder ligero que integra estas características para producir una segmentación precisa.
Todo esto sin necesidad de embeddings posicionales.

Algunas características destacadas de SegFormer:

- Al no trabajar con embeddings posicionales, puede manejar imágenes de tamaños variados sin pérdida de información espacial.  
- Arquitectura simple pero potente, con un balance óptimo entre número de parámetros y velocidad.  
- Entrenado sobre datasets estándares como ADE20k, logrando resultados de punta en benchmarks públicos al momento de su publicación.

Para más detalles técnicos y resultados, puedes consultar el [paper original de SegFormer](https://arxiv.org/abs/2105.15203).

## Algunos ejemplos de segmentación semántica con SegFormer
""")

segformer_examples = [
    "data/images/001_segformer_img.png",
    "data/images/002_segformer_img.png",
    "data/images/003_segformer_img.png"
]

captions = [
    "Segmentación Semántica - Moto en el aire",
    "Segmentación Semántica - Jugadores en el campo de béisbol",
    "Segmentación Semántica - Surfistas en la playa"
    ]

# Inicializar el índice en session_state
if "carousel_index" not in st.session_state:
    st.session_state.carousel_index = 0

# Crear layout con 3 columnas
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    if st.button("◀", use_container_width=True) and st.session_state.carousel_index > 0:
        st.session_state.carousel_index -= 1

with col3:
    if st.button("▶", use_container_width=True) and st.session_state.carousel_index < len(segformer_examples) - 1:
        st.session_state.carousel_index += 1

# Mostrar la imagen en el centro
with col2:
    current_idx = st.session_state.carousel_index
    if os.path.exists(segformer_examples[current_idx]):
        st.image(segformer_examples[current_idx], caption=captions[current_idx], use_container_width=True)
    else:
        st.warning(f"No se encontró la imagen: {segformer_examples[current_idx]}")

st.markdown("""
## Arquitectura del SegFormer """)

st.image("data/images/000_Architecture.png", caption="Arquitectura SegFormer (Encoder + Decoder)", width=800)

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
