# src/01_Explicacion.py
import streamlit as st

st.set_page_config(page_title="1. Segmentación", page_icon="🧠", layout="wide")

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
    width=900,
)

st.markdown(
    """
La segmentación semántica es una tarea fundamental en visión por computadora que consiste en clasificar cada píxel de una imagen asignándole una etiqueta que indica el objeto o categoría a la que pertenece.  
Esto permite entender la escena a nivel granular, diferenciando objetos y regiones con significado semántico.

### Entradas y salidas

- **Entrada:** Imagen a color en formato RGB, con resolución arbitraria.  
- **Salida:** Máscara semántica, es decir, una matriz bidimensional donde cada píxel tiene un valor entero que representa una clase específica del dataset.

### ¿Por qué es importante esta tarea?

La segmentación semántica es crítica para aplicaciones como conducción autónoma, robótica, realidad aumentada, análisis médico, y más, ya que provee un entendimiento detallado y localizado de la escena visual.

### El modelo SegFormer

El modelo empleado es **SegFormer**, una arquitectura reciente basada en Transformers diseñada específicamente para segmentación semántica con eficiencia y precisión.  
SegFormer combina un encoder jerárquico eficiente que captura características a múltiples escalas y un decoder ligero que integra estas características para producir una segmentación precisa.

Algunas características destacadas de SegFormer:

- No utiliza embeddings posicionales explícitos, lo que permite trabajar con imágenes de tamaños variados sin pérdida de información espacial.  
- Arquitectura simple pero potente, con un balance óptimo entre número de parámetros y velocidad.  
- Entrenado sobre datasets estándares como ADE20k, logrando resultados de punta en benchmarks públicos.

Para más detalles técnicos y resultados, puedes consultar el [paper original de SegFormer](https://arxiv.org/abs/2105.15203).
"""
)

try:
    st.image("data/images/001_segformer_img.png", caption="Ejemplo de segmentación semántica")
except FileNotFoundError:
    st.warning("⚠️ No se pudo cargar la imagen. Verifica la ruta: data/images/001_segformer_img.png")
