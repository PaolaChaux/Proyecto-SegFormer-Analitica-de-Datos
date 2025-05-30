# 03_Inferencia.py
import streamlit as st
from PIL import Image
import numpy as np

from segformer_model import (
    load_model, get_classes, segment_image,
    mostrar_segmentacion_con_leyenda
)

st.set_page_config(
    page_title="Inferencia",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Esta app fue creada por Marce para demo de Segmentación Semántica."
    }
)

# Cargar estilos CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('src/styles.css')

# Header principal con hero section
st.markdown(
    """
    <div class="hero-section">
        <div class="hero-content">
            <div class="hero-icon">🤖</div>
            <h1 class="hero-title">Inferencia en Tiempo Real</h1>
            <p class="hero-subtitle">Segmentación semántica desde archivos</p>
            <div class="hero-badges">
                <div class="badge">Tiempo Real</div>
                <div class="badge">SegFormer</div>
                <div class="badge">CPU/GPU</div>
            </div>
        </div>
        <div class="hero-decoration"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Contenedor principal para el contenido
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Cargar modelo con caché para evitar recarga
@st.cache_resource
def get_model():
    return load_model()

processor, model, device = get_model()
classes = get_classes(model)

# Sección de información del modelo
st.markdown(
    """
    <div class="demo-section">
        <div class="content-card tech-card">
            <div class="card-header">
                <div class="tech-icon">⚙️</div>
                <h2>Modelo Cargado</h2>
            </div>
            <div class="card-content">
                <p><strong>Dispositivo:</strong> <code>{}</code></p>
                <p><strong>Clases disponibles:</strong> {} categorías</p>
                <p><strong>Estado:</strong> ✅ Listo para inferencia</p>
            </div>
        </div>
    </div>
    """.format(device, len(classes)),
    unsafe_allow_html=True,
)

# Sección carga archivo
st.markdown(
    """
    <div class="demo-section">
        <div class="section-header">
            <h2>📁 Inferencia desde Archivo</h2>
            <p>Sube una imagen para procesamiento</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="content-card">
        <div class="card-header">
            <h2>Cargar Archivo</h2>
        </div>
        <div class="card-content">
    """,
    unsafe_allow_html=True,
)

file = st.file_uploader(
    "Selecciona un archivo para procesar", 
    type=["jpg", "png", "jpeg"],
    help="Formatos soportados: JPG, PNG, JPEG para imágenes"
)

st.markdown('</div></div>', unsafe_allow_html=True)

if file:
    # Procesamiento de imagen
    st.markdown(
        """
        <div class="content-card highlight">
            <div class="card-header">
                <h2>🖼️ Procesando Imagen</h2>
            </div>
            <div class="card-content">
        """,
        unsafe_allow_html=True,
    )
    
    image = Image.open(file).convert("RGB")
    
    # Definimos altura fija para las imágenes (por ejemplo, 400px)
    fixed_height = 400

    # Obtener ancho proporcional para la imagen original
    orig_w, orig_h = image.size
    new_width = int((fixed_height / orig_h) * orig_w)

    # Procesar máscara con tamaño fijo para que coincida la altura
    mask, duration = segment_image(image, processor, model, device)
    mask_resized_image = image.resize((new_width, fixed_height))

    # Columnas para las imágenes
    col1, col2 = st.columns([1,1], gap="medium")

    with col1:
        st.markdown("<div style='width: 100%; overflow: hidden; text-align: center; margin-bottom: 0;'>", unsafe_allow_html=True)
        st.image(image, caption="📤 Imagen Original", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='width: 100%; overflow: hidden; text-align: center; margin-bottom: 0;'>", unsafe_allow_html=True)
        mostrar_segmentacion_con_leyenda(mask_resized_image, mask, classes, st)
        st.markdown("</div>", unsafe_allow_html=True)

    # Cuadro con clases detectadas horizontal y centrado debajo
    detected_idx = np.unique(mask)
    class_list = [classes[i] for i in detected_idx]
    st.markdown(
        f"""
        <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0; text-align: center;">
            <strong>🏷️ Clases detectadas:</strong> {", ".join(class_list)}
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.markdown('</div></div>', unsafe_allow_html=True)

# Cerrar contenedor principal
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <div class="footer">
        <p>🤖 Inferencia en Tiempo Real - SegFormer</p>
    </div>
    """,
    unsafe_allow_html=True,
)