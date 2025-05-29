import streamlit as st

st.set_page_config(
    page_title="Segmentación Semántica", 
    page_icon="🖥️", 
    layout="wide")

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
            <div class="hero-icon">🖥️</div>
            <h1 class="hero-title">¿Qué es la segmentación semántica?</h1>
            <p class="hero-subtitle">Santiago Murgueitio - Paola Chaux - Sebastian Ortiz</p>
        </div>
        <div class="hero-decoration"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Contenedor principal para el contenido
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Sección de demostración visual
st.markdown(
    """
    <div class="demo-section">
        <div class="section-header">
            <h2>Demostración Visual</h2>
            <p>Observa cómo funciona la segmentación semántica en tiempo real</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Contenedor para la imagen GIF
st.image(
    "https://github.com/NVlabs/SegFormer/blob/master/resources/seg_demo.gif?raw=true",
    caption="Segmentación Semántica en Acción",
    width=800,
)
st.markdown('</div>', unsafe_allow_html=True)

# Sección de definición
st.markdown(
    """
    <div class="definition-section">
        <div class="content-card">
            <div class="card-header">
                <h2>¿Qué es?</h2>
            </div>
            <div class="card-content">
                <p>La segmentación semántica es una tarea fundamental en visión por computadora que consiste en clasificar cada píxel de una imagen asignándole una etiqueta que indica el objeto o categoría a la que pertenece.</p>
                <p>Esto permite entender la escena a nivel granular, diferenciando objetos y regiones con significado semántico.</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sección de importancia
st.markdown(
    """
    <div class="importance-section">
        <div class="content-card highlight">
            <div class="card-header">
                <h2>¿Por qué es importante?</h2>
            </div>
            <div class="card-content">
                <p>La segmentación semántica es una de las tareas más complejas en la visión por computadora, ya que requiere no solo identificar objetos,
                 sino también entender su contexto y relaciones espaciales, de forma muy detallada.
                 Su uso es elemental para aplicaciones como: </p>
                <div class="applications-grid">
                    <div class="app-item">
                        <div class="app-icon">🚗</div>
                        <span>Conducción autónoma</span>
                    </div>
                    <div class="app-item">
                        <div class="app-icon">🤖</div>
                        <span>Robótica</span>
                    </div>
                    <div class="app-item">
                        <div class="app-icon">👁️</div>
                        <span>Realidad aumentada</span>
                    </div>
                    <div class="app-item">
                        <div class="app-icon">🏥</div>
                        <span>Análisis médico</span>
                    </div>
                    <div class="app-item">
                        <div class="app-icon">🗺️</div>
                        <span>Análisis Geoespacial</span>
                    </div>
                    <div class="app-item">
                        <div class="app-icon">🔍</div>
                        <span>Investigación</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sección de funcionamiento
st.markdown(
    """
    <div class="how-it-works-section">
        <div class="content-card">
            <div class="card-header">
                <h2>¿Cómo funciona?</h2>
            </div>
            <div class="card-content">
                <p>En la segmentación clásica, cada píxel de la imagen se clasifica en una de varias categorías predefinidas. Estos píxeles pueden representar objetos como personas, vehículos, edificios, etc.
                Como este proceso se realiza a nivel de pixel esto nos permite tener bordes claros para los diferentes objetos</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


try:
    st.image("data/images/001_segformer_img.png", caption="Ejemplo de segmentación semántica", width=800)
except FileNotFoundError:
    st.markdown(
        """
        <div class="warning-box">
            <div class="warning-icon">⚠️</div>
            <div class="warning-text">
                <strong>Imagen no encontrada</strong><br>
                No se pudo cargar la imagen. Verifica la ruta: data/images/001_segformer_img.png
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown('</div>', unsafe_allow_html=True)

# Sección de tareas de visión por computadora
st.markdown(
    """
    <div class="tasks-section">
        <div class="content-card">
            <div class="card-header">
                <h2>Tareas comunes de visión por computadora</h2>
            </div>
            <div class="card-content">
                <p>A continuación, podemos observar de manera gráfica la diferencia de la segmentación semantica con respecto a otras tecnicas de visión por computadora</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


try:
    st.image("data/images/computer-vision.png", caption="Tareas de visión por computadora", width=800)
except FileNotFoundError:
    st.markdown(
        """
        <div class="warning-box">
            <div class="warning-icon">⚠️</div>
            <div class="warning-text">
                <strong>Imagen no encontrada</strong><br>
                No se pudo cargar la imagen. Verifica la ruta: data/images/computer-vision.png
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown('</div>', unsafe_allow_html=True)

# Cerrar contenedor principal
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <div class="footer">
        <p>🖥️ Segmentación Semántica - Visión por Computadora</p>
    </div>
    """,
    unsafe_allow_html=True,
)