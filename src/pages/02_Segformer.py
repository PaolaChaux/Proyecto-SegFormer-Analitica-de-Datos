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

# Cargar estilos CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css('src/styles.css')

# Hero Section unificada
st.markdown(
    """
    <div class="hero-section">
        <div class="hero-content">
            <div class="hero-icon">🔍</div>
            <h1 class="hero-title">SegFormer</h1>
            <p class="hero-subtitle">Arquitectura Transformer para Segmentación Semántica</p>
        </div>
        <div class="hero-decoration"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Contenedor principal
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Introducción
st.markdown(
    """
    <div class="content-card">
        <div class="card-header">
            <h2>¿Qué es SegFormer?</h2>
        </div>
        <div class="card-content">
            <p>Modelo de <strong>NVIDIA</strong> basado en <strong>Transformers</strong> para segmentación semántica, combinando eficiencia y precisión.</p>
            <p>Utiliza un encoder jerárquico sin embeddings posicionales y un decoder ligero para segmentar imágenes de forma precisa y rápida.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Características
st.markdown(
    """
    <div class="content-card">
        <div class="card-header">
            <h2>Características Destacadas</h2>
        </div>
        <div class="card-content">
            <ul>
                <li><strong>Sin Embeddings Posicionales:</strong> admite tamaños variados sin perder precisión espacial.</li>
                <li><strong>Arquitectura Eficiente:</strong> excelente balance entre velocidad y parámetros.</li>
                <li><strong>Resultados Sólidos:</strong> entrenado en ADE20k con excelente desempeño.</li>
            </ul>
            <a class="paper-button" href="https://arxiv.org/abs/2105.15203" target="_blank">📄 Ver Paper Original</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Ejemplos
st.markdown(
    """
    <div class="section-header">
        <h2>Ejemplos de Segmentación Semántica</h2>
        <p>Observa los resultados de SegFormer en diferentes escenarios</p>
    </div>
    """,
    unsafe_allow_html=True,
)

segformer_examples = [
    "data/images/001_segformer_img.png",
    "data/images/002_segformer_img.png",
    "data/images/003_segformer_img.png"
]

captions = [
    "Segmentación - Moto en el aire",
    "Segmentación - Jugadores en el campo",
    "Segmentación - Surfistas en la playa"
]

if "carousel_index" not in st.session_state:
    st.session_state.carousel_index = 0

col1, col2, col3 = st.columns([1, 8, 1])

with col1:
    if st.button("◀", key="prev"):
        if st.session_state.carousel_index > 0:
            st.session_state.carousel_index -= 1

with col3:
    if st.button("▶", key="next"):
        if st.session_state.carousel_index < len(segformer_examples) - 1:
            st.session_state.carousel_index += 1

with col2:
    idx = st.session_state.carousel_index
    if os.path.exists(segformer_examples[idx]):
        st.image(segformer_examples[idx], caption=captions[idx], use_container_width=True)
    else:
        st.warning(f"No se encontró la imagen: {segformer_examples[idx]}")

# Arquitectura
st.markdown(
    """
    <div class="content-card">
        <div class="card-header">
            <h2>Arquitectura del SegFormer</h2>
        </div>
        <div class="card-content">
            <p>Encoder jerárquico con múltiples escalas y decoder ligero MLP.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    st.image("data/images/000_Architecture.png", caption="Arquitectura SegFormer", use_container_width=True)
except FileNotFoundError:
    st.warning("No se encontró la imagen de arquitectura.")

# Encoder
st.markdown(
    """
    <div class="content-card">
        <div class="card-header">
            <h2>Encoder: Mix Transformer (MiT)</h2>
        </div>
        <div class="card-content">
            <ul>
                <li>Jerárquico, inspirado en Swin Transformer</li>
                <li>Procesamiento multiresolución: <code>1/4, 1/8, 1/16, 1/32</code></li>
                <li>Atención local y global</li>
                <li>Versión MiT-B0 tiene 3.7M parámetros</li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Decoder
st.markdown(
    """
    <div class="content-card">
        <div class="card-header">
            <h2>Decoder: MLP Eficiente</h2>
        </div>
        <div class="card-content">
            <ul>
                <li>Proyecciones y fusión de características multiescala</li>
                <li>MLP simple y rápido</li>
                <li>0.4M parámetros</li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Ventajas
st.markdown(
    """
    <div class="content-card">
        <div class="card-header">
            <h2>Ventajas Clave</h2>
        </div>
        <div class="card-content">
            <ul>
                <li>🚫 Sin Positional Embeddings</li>
                <li>📊 Jerarquía eficiente para contexto y detalle</li>
                <li>⚡ Inferencia rápida: hasta 47 FPS</li>
                <li>🔧 Modular: MiT-B0 a B5</li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Parámetros
st.markdown(
    """
    <div class="content-card">
        <div class="card-header">
            <h2>Parámetros (MiT-B0)</h2>
        </div>
        <div class="card-content">
            <table>
                <thead><tr><th>Componente</th><th>Parámetros (M)</th></tr></thead>
                <tbody>
                    <tr><td>Encoder</td><td>3.7</td></tr>
                    <tr><td>Decoder</td><td>0.4</td></tr>
                    <tr><td><strong>Total</strong></td><td><strong>~4.1</strong></td></tr>
                </tbody>
            </table>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Referencias
st.markdown(
    """
    <div class="content-card">
        <div class="card-header">
            <h2>Referencias</h2>
        </div>
        <div class="card-content">
            <p><strong>Xie et al.</strong>, "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers", NeurIPS 2021</p>
            <a href="https://arxiv.org/abs/2105.15203" target="_blank">📄 Paper en arXiv</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Footer
st.markdown(
    """
    <div class="footer">
        <p>🔍 SegFormer - Segmentación Semántica con Transformers</p>
    </div>
    """,
    unsafe_allow_html=True,
)
