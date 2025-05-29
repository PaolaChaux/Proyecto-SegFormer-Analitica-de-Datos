import streamlit as st

st.set_page_config(
    page_title="Segmentación Semántica", 
    page_icon="🧠", 
    layout="wide")


with st.container():
    st.markdown(
        """
        <div class="header-box">
            <h1>¿Qué es la segmentación semántica?</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.image(
    "https://github.com/NVlabs/SegFormer/blob/master/resources/seg_demo.gif?raw=true",
    caption="Segmentación Semántica",
    width=700,
)

st.markdown(
    """
La segmentación semántica es una tarea fundamental en visión por computadora que consiste en clasificar cada píxel de una imagen asignándole una etiqueta que indica el objeto o categoría a la que pertenece.  
Esto permite entender la escena a nivel granular, diferenciando objetos y regiones con significado semántico.

### ¿Por qué es importante esta tarea?

La segmentación semántica es crítica para aplicaciones como conducción autónoma, robótica, realidad aumentada, análisis médico, y más, ya que provee un entendimiento detallado y localizado de la escena visual.

### ¿Cómo funciona?

En la segmentación clásica, cada píxel de la imagen se clasifica en una de varias categorías predefinidas.
estos pixeles pueden representar objetos como personas, vehículos, edificios, etc.
"""
)

try:
    st.image("data/images/001_segformer_img.png", caption="Ejemplo de segmentación semántica", width=700)
except FileNotFoundError:
    st.warning("⚠️ No se pudo cargar la imagen. Verifica la ruta: data/images/001_segformer_img.png")

st.markdown(
    """
### Tareas comunes de visión por computadora
"""
)

try:
    st.image("data/images/computer-vision.png", caption="Ejemplo de segmentación semántica", width=700)
except FileNotFoundError:
    st.warning("⚠️ No se pudo cargar la imagen. Verifica la ruta: data/images/001_segformer_img.png")