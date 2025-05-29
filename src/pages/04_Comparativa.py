# 04_Comparativa.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np

st.set_page_config(
    page_title="Comparativa Inferencia", 
    page_icon="📊", 
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
            <div class="hero-icon">📊</div>
            <h1 class="hero-title">Comparativa de Rendimiento</h1>
            <p class="hero-subtitle">Análisis de tiempos de inferencia por hardware y video</p>
            <div class="hero-badges">
                <div class="badge">RTX 3050</div>
                <div class="badge">GTX 1650</div>
                <div class="badge">Benchmarks</div>
            </div>
        </div>
        <div class="hero-decoration"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Contenedor principal para el contenido
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Datos de prueba
data = {
    "Video": [
        "video1",
        "video2", 
        "video3"
    ],
    "Resolución": ["640x360 px", "2560x1440 px", "1080x1920 px"],
    "Frames": [249, 341, 596],
    "Tamaño": ["495.9 KB", "25.5 MB", "12.9 MB"],
    "Duración": ["~8.3 s", "~11.4 s", "~19.9 s"],
    "RTX 3050, 4 GB VRAM": [
        "30.15s total (0.047s/frame)",
        "63.92s total (0.045s/frame)",
        "83.84s total (0.045s/frame)"
    ],
    "GTX 1650, 4 GB VRAM": [
        "62.02s total (0.094s/frame)",
        "95.02s total (0.100s/frame)",
        "145.41s total (0.091s/frame)"
    ]
}

df = pd.DataFrame(data)

# Sección de tabla comparativa
st.markdown(
    """
    <div class="demo-section">
        <div class="section-header">
            <h2>📋 Tabla Comparativa Detallada</h2>
            <p>Resultados de inferencia en diferentes configuraciones de hardware</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="content-card">
        <div class="card-header">
            <h2>Resultados de Benchmarks</h2>
        </div>
        <div class="card-content">
            <div class="parameters-table">
    """,
    unsafe_allow_html=True,
)

st.dataframe(df, use_container_width=True)

st.markdown('</div></div></div>', unsafe_allow_html=True)

# Análisis de rendimiento
st.markdown(
    """
    <div class="demo-section">
        <div class="section-header">
            <h2>📈 Análisis de Rendimiento</h2>
            <p>Comparación visual de tiempos de procesamiento</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Extraer tiempos totales como floats
def parse_time(s):
    return float(s.split('s')[0])

def parse_time_per_frame(s):
    return float(s.split('(')[1].split('s/frame')[0])

times_rtx = [parse_time(t) for t in df["RTX 3050, 4 GB VRAM"]]
times_gtx = [parse_time(t) for t in df["GTX 1650, 4 GB VRAM"]]
times_rtx_frame = [parse_time_per_frame(t) for t in df["RTX 3050, 4 GB VRAM"]]
times_gtx_frame = [parse_time_per_frame(t) for t in df["GTX 1650, 4 GB VRAM"]]

# Crear gráficos
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="content-card">
            <div class="card-header">
                <h2>⏱️ Tiempo Total de Procesamiento</h2>
            </div>
            <div class="card-content">
        """,
        unsafe_allow_html=True,
    )
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    indices = range(len(df))
    width = 0.35

    bars1 = ax1.bar([i - width/2 for i in indices], times_rtx, width, 
                    label="RTX 3050", color='#667eea', alpha=0.8)
    bars2 = ax1.bar([i + width/2 for i in indices], times_gtx, width, 
                    label="GTX 1650", color='#764ba2', alpha=0.8)

    ax1.set_xticks(indices)
    ax1.set_xticklabels(df["Video"])
    ax1.set_ylabel("Tiempo total (segundos)")
    ax1.set_title("Comparación de Tiempo Total")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Añadir valores en las barras
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')

    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", bbox_inches='tight', dpi=150, 
                facecolor='white', edgecolor='none')
    buf1.seek(0)
    
    st.image(buf1, use_column_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

with col2:
    st.markdown(
        """
        <div class="content-card">
            <div class="card-header">
                <h2>🚀 Tiempo por Frame</h2>
            </div>
            <div class="card-content">
        """,
        unsafe_allow_html=True,
    )
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    bars3 = ax2.bar([i - width/2 for i in indices], times_rtx_frame, width, 
                    label="RTX 3050", color='#667eea', alpha=0.8)
    bars4 = ax2.bar([i + width/2 for i in indices], times_gtx_frame, width, 
                    label="GTX 1650", color='#764ba2', alpha=0.8)

    ax2.set_xticks(indices)
    ax2.set_xticklabels(df["Video"])
    ax2.set_ylabel("Tiempo por frame (segundos)")
    ax2.set_title("Comparación de Tiempo por Frame")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Añadir valores en las barras
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.3f}s', ha='center', va='bottom', fontweight='bold')

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", bbox_inches='tight', dpi=150, 
                facecolor='white', edgecolor='none')
    buf2.seek(0)
    
    st.image(buf2, use_column_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

# Métricas de rendimiento
st.markdown(
    """
    <div class="demo-section">
        <div class="section-header">
            <h2>⚡ Métricas de Rendimiento</h2>
            <p>Estadísticas clave del rendimiento de inferencia</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Calcular estadísticas
rtx_avg = np.mean(times_rtx_frame)
gtx_avg = np.mean(times_gtx_frame)
speed_improvement = ((gtx_avg - rtx_avg) / gtx_avg) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="content-card highlight">
            <div class="card-content" style="text-align: center; padding: 2rem;">
                <h3 style="color: white; margin-bottom: 0.5rem;">RTX 3050</h3>
                <div style="font-size: 2rem; font-weight: bold; color: white;">{:.3f}s</div>
                <p style="color: rgba(255,255,255,0.9);">Promedio por frame</p>
            </div>
        </div>
        """.format(rtx_avg),
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="content-card">
            <div class="card-content" style="text-align: center; padding: 2rem;">
                <h3 style="color: var(--text-color); margin-bottom: 0.5rem;">GTX 1650</h3>
                <div style="font-size: 2rem; font-weight: bold; color: var(--primary-color);">{:.3f}s</div>
                <p style="color: var(--text-light);">Frames por segundo</p>
            </div>
        </div>
        """.format(fps_rtx),
        unsafe_allow_html=True,
    )

# Análisis detallado por hardware
st.markdown(
    """
    <div class="demo-section">
        <div class="section-header">
            <h2>🔍 Análisis Detallado por Hardware</h2>
            <p>Comparación exhaustiva de las capacidades de cada GPU</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="content-card tech-card">
            <div class="card-header">
                <div class="tech-icon">🎮</div>
                <h2>NVIDIA RTX 3050</h2>
            </div>
            <div class="card-content">
                <ul class="tech-list">
                    <li><strong>VRAM:</strong> <code>4 GB GDDR6</code></li>
                    <li><strong>CUDA Cores:</strong> <code>2,560</code></li>
                    <li><strong>Tensor Cores:</strong> <code>Generación 3</code></li>
                    <li><strong>Tiempo promedio:</strong> <code>{:.3f}s/frame</code></li>
                    <li><strong>Rendimiento:</strong> <code>{:.1f} FPS</code></li>
                    <li><strong>Ventaja:</strong> Arquitectura moderna con Tensor Cores</li>
                </ul>
            </div>
        </div>
        """.format(rtx_avg, fps_rtx),
        unsafe_allow_html=True,
    )

with col2:
    fps_gtx = 1 / gtx_avg
    st.markdown(
        """
        <div class="content-card tech-card">
            <div class="card-header">
                <div class="tech-icon">⚙️</div>
                <h2>NVIDIA GTX 1650</h2>
            </div>
            <div class="card-content">
                <ul class="tech-list">
                    <li><strong>VRAM:</strong> <code>4 GB GDDR5</code></li>
                    <li><strong>CUDA Cores:</strong> <code>896</code></li>
                    <li><strong>Tensor Cores:</strong> <code>No disponible</code></li>
                    <li><strong>Tiempo promedio:</strong> <code>{:.3f}s/frame</code></li>
                    <li><strong>Rendimiento:</strong> <code>{:.1f} FPS</code></li>
                    <li><strong>Limitación:</strong> Arquitectura anterior sin optimizaciones AI</li>
                </ul>
            </div>
        </div>
        """.format(gtx_avg, fps_gtx),
        unsafe_allow_html=True,
    )

# Factores que afectan el rendimiento
st.markdown(
    """
    <div class="demo-section">
        <div class="content-card highlight">
            <div class="card-header">
                <h2>🔧 Factores que Afectan el Rendimiento</h2>
            </div>
            <div class="card-content">
                <div class="features-grid">
                    <div class="feature-item">
                        <div class="feature-icon">📐</div>
                        <div class="feature-content">
                            <h3>Resolución</h3>
                            <p>Mayor resolución = más píxeles para procesar. Videos 4K requieren significativamente más tiempo que HD.</p>
                        </div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">🧠</div>
                        <div class="feature-content">
                            <h3>Arquitectura GPU</h3>
                            <p>GPUs modernas con Tensor Cores optimizan operaciones de inferencia AI, reduciendo tiempos considerablemente.</p>
                        </div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">💾</div>
                        <div class="feature-content">
                            <h3>VRAM Disponible</h3>
                            <p>Modelos más grandes requieren más memoria. 4GB es el mínimo recomendado para SegFormer.</p>
                        </div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">⚡</div>
                        <div class="feature-content">
                            <h3>Optimizaciones</h3>
                            <p>TensorRT, FP16, y batch processing pueden acelerar significativamente la inferencia.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Recomendaciones
st.markdown(
    """
    <div class="demo-section">
        <div class="content-card">
            <div class="card-header">
                <h2>💡 Recomendaciones de Hardware</h2>
            </div>
            <div class="card-content">
                <p>Basado en el análisis de rendimiento, sugerimos considerar GPUs RTX para tareas de segmentación semántica intensivas.</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Análisis detallado por hardware
st.markdown(
    """
    <div class="demo-section">
        <div class="section-header">
            <h2>🔍 Análisis Detallado por Hardware</h2>
            <p>Comparación exhaustiva de las capacidades de cada GPU</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="content-card tech-card">
            <div class="card-header">
                <div class="tech-icon">🎮</div>
                <h2>NVIDIA RTX 3050</h2>
            </div>
            <div class="card-content">
                <ul class="tech-list">
                    <li><strong>VRAM:</strong> <code>4 GB GDDR6</code></li>
                    <li><strong>CUDA Cores:</strong> <code>2,560</code></li>
                    <li><strong>Tensor Cores:</strong> <code>Generación 3</code></li>
                    <li><strong>Tiempo promedio:</strong> <code>{:.3f}s/frame</code></li>
                    <li><strong>Rendimiento:</strong> <code>{:.1f} FPS</code></li>
                    <li><strong>Ventaja:</strong> Arquitectura moderna con Tensor Cores</li>
                </ul>
            </div>
        </div>
        """.format(rtx_avg, fps_rtx),
        unsafe_allow_html=True,
    )

with col2:
    fps_gtx = 1 / gtx_avg
    st.markdown(
        """
        <div class="content-card tech-card">
            <div class="card-header">
                <div class="tech-icon">⚙️</div>
                <h2>NVIDIA GTX 1650</h2>
            </div>
            <div class="card-content">
                <ul class="tech-list">
                    <li><strong>VRAM:</strong> <code>4 GB GDDR5</code></li>
                    <li><strong>CUDA Cores:</strong> <code>896</code></li>
                    <li><strong>Tensor Cores:</strong> <code>No disponible</code></li>
                    <li><strong>Tiempo promedio:</strong> <code>{:.3f}s/frame</code></li>
                    <li><strong>Rendimiento:</strong> <code>{:.1f} FPS</code></li>
                    <li><strong>Limitación:</strong> Arquitectura anterior sin optimizaciones AI</li>
                </ul>
            </div>
        </div>
        """.format(gtx_avg, fps_gtx),
        unsafe_allow_html=True,
    )

# Factores que afectan el rendimiento
st.markdown(
    """
    <div class="demo-section">
        <div class="content-card highlight">
            <div class="card-header">
                <h2>🔧 Factores que Afectan el Rendimiento</h2>
            </div>
            <div class="card-content">
                <div class="features-grid">
                    <div class="feature-item">
                        <div class="feature-icon">📐</div>
                        <div class="feature-content">
                            <h3>Resolución</h3>
                            <p>Mayor resolución = más píxeles para procesar. Videos 4K requieren significativamente más tiempo que HD.</p>
                        </div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">🧠</div>
                        <div class="feature-content">
                            <h3>Arquitectura GPU</h3>
                            <p>GPUs modernas con Tensor Cores optimizan operaciones de inferencia AI, reduciendo tiempos considerablemente.</p>
                        </div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">💾</div>
                        <div class="feature-content">
                            <h3>VRAM Disponible</h3>
                            <p>Modelos más grandes requieren más memoria. 4GB es el mínimo recomendado para SegFormer.</p>
                        </div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">⚡</div>
                        <div class="feature-content">
                            <h3>Optimizaciones</h3>
                            <p>TensorRT, FP16, y batch processing pueden acelerar significativamente la inferencia.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Recomendaciones
st.markdown(
    """
    <div class="demo-section">
        <div class="content-card">
            <div class="card-header">
                <h2>💡 Recomendaciones de Hardware</h2>
            </div>
            <div class="card-content">
                <div class="advantages-grid">
                    <div class="advantage-item">
                        <div class="advantage-icon">🟢</div>
                        <div class="advantage-content">
                            <h3>Tiempo Real (>30 FPS)</h3>
                            <p>RTX 3060 o superior recomendado para aplicaciones que requieren procesamiento en tiempo real.</p>
                        </div>
                    </div>
                    <div class="advantage-item">
                        <div class="advantage-icon">🟡</div>
                        <div class="advantage-content">
                            <h3>Procesamiento Batch</h3>
                            <p>RTX 3050 o GTX 1660 son suficientes para procesamiento offline de videos.</p>
                        </div>
                    </div>
                    <div class="advantage-item">
                        <div class="advantage-icon">🔴</div>
                        <div class="advantage-content">
                            <h3>Presupuesto Limitado</h3>
                            <p>GTX 1650 funciona pero con tiempos de procesamiento más largos. Considera CPU para casos simples.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabla de especificaciones técnicas
st.markdown(
    """
    <div class="demo-section">
        <div class="section-header">
            <h2>📋 Especificaciones Técnicas Completas</h2>
            <p>Comparación detallada de las características de hardware</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="content-card">
        <div class="card-header">
            <h2>Especificaciones de GPU</h2>
        </div>
        <div class="card-content">
            <div class="parameters-table">
    """,
    unsafe_allow_html=True,
)

specs_data = {
    "Especificación": [
        "Arquitectura",
        "Proceso de Fabricación",
        "CUDA Cores",
        "Tensor Cores",
        "VRAM",
        "Ancho de Banda Memoria",
        "TDP",
        "Precio Aprox. (USD)",
        "Tiempo Promedio/Frame",
        "FPS Teórico",
        "Eficiencia (FPS/Watt)"
    ],
    "RTX 3050": [
        "Ampere",
        "8nm (Samsung)",
        "2,560",
        "80 (3ra Gen)",
        "8 GB GDDR6",
        "224 GB/s",
        "130W",
        "$249",
        f"{rtx_avg:.3f}s",
        f"{fps_rtx:.1f}",
        f"{fps_rtx/130:.3f}"
    ],
    "GTX 1650": [
        "Turing",
        "12nm (TSMC)",
        "896",
        "No",
        "4 GB GDDR5",
        "128 GB/s",
        "75W",
        "$149",
        f"{gtx_avg:.3f}s",
        f"{fps_gtx:.1f}",
        f"{fps_gtx/75:.3f}"
    ]
}

specs_df = pd.DataFrame(specs_data)
st.dataframe(specs_df, use_container_width=True)

st.markdown('</div></div></div>', unsafe_allow_html=True)

# Conclusiones
st.markdown(
    """
    <div class="demo-section">
        <div class="content-card highlight">
            <div class="card-header">
                <h2>🎯 Conclusiones del Benchmark</h2>
            </div>
            <div class="card-content">
                <p><strong>🏆 Ganador claro:</strong> La RTX 3050 supera consistentemente a la GTX 1650 con un <strong>{:.1f}% de mejora</strong> en velocidad promedio.</p>
                <p><strong>🔬 Factor clave:</strong> Los Tensor Cores de tercera generación en la RTX 3050 proporcionan aceleración específica para cargas de trabajo de AI.</p>
                <p><strong>💰 Relación precio/rendimiento:</strong> Aunque la RTX 3050 cuesta ~67% más, ofrece {:.1f}x mejor rendimiento, justificando la inversión para aplicaciones AI.</p>
                <p><strong>⚡ Recomendación:</strong> Para segmentación semántica en producción, la RTX 3050 es la opción mínima recomendada para workflows eficientes.</p>
            </div>
        </div>
    </div>
    """.format(speed_improvement, fps_rtx/fps_gtx),
    unsafe_allow_html=True,
)

# Cerrar contenedor principal
st.markdown('</div>', unsafe_allow_html=True)
# Recomendaciones gráficas (Advantage Items)
st.markdown(
    """
    <div class="demo-section">
        <div class="content-card">
            <div class="card-header">
                <h2>📈 Ventajas según uso</h2>
            </div>
            <div class="card-content">
                <div class="advantage-item">
                    <div class="advantage-icon">🟢</div>
                    <div class="advantage-content">
                        <h3>Tiempo Real (>30 FPS)</h3>
                        <p>RTX 3060 o superior recomendado para aplicaciones que requieren procesamiento en tiempo real.</p>
                    </div>
                </div>
                <div class="advantage-item">
                    <div class="advantage-icon">🟡</div>
                    <div class="advantage-content">
                        <h3>Procesamiento Batch</h3>
                        <p>RTX 3050 o GTX 1660 son suficientes para procesamiento offline de videos.</p>
                    </div>
                </div>
                <div class="advantage-item">
                    <div class="advantage-icon">🔴</div>
                    <div class="advantage-content">
                        <h3>Presupuesto Limitado</h3>
                        <p>GTX 1650 funciona pero con tiempos de procesamiento más largos. Considera CPU para casos simples.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

specs_data = {
    "Especificación": [
        "Arquitectura",
        "Proceso de Fabricación",
        "CUDA Cores",
        "Tensor Cores",
        "VRAM",
        "Ancho de Banda Memoria",
        "TDP",
        "Precio Aprox. (USD)",
        "Tiempo Promedio/Frame",
        "FPS Teórico",
        "Eficiencia (FPS/Watt)"
    ],
    "RTX 3050": [
        "Ampere",
        "8nm (Samsung)",
        "2,560",
        "80 (3ra Gen)",
        "8 GB GDDR6",
        "224 GB/s",
        "130W",
        "$249",
        f"{rtx_avg:.3f}s",
        f"{fps_rtx:.1f}",
        f"{fps_rtx/130:.3f}"
    ],
    "GTX 1650": [
        "Turing",
        "12nm (TSMC)",
        "896",
        "No",
        "4 GB GDDR5",
        "128 GB/s",
        "75W",
        "$149",
        f"{gtx_avg:.3f}s",
        f"{fps_gtx:.1f}",
        f"{fps_gtx/75:.3f}"
    ]
}

specs_df = pd.DataFrame(specs_data)
st.dataframe(specs_df, use_container_width=True)

st.markdown('</div></div></div>', unsafe_allow_html=True)

# Conclusiones
st.markdown(
    """
    <div class="demo-section">
        <div class="content-card highlight">
            <div class="card-header">
                <h2>🎯 Conclusiones del Benchmark</h2>
            </div>
            <div class="card-content">
                <p><strong>🏆 Ganador claro:</strong> La RTX 3050 supera consistentemente a la GTX 1650 con un <strong>{:.1f}% de mejora</strong> en velocidad promedio.</p>
                <p><strong>🔬 Factor clave:</strong> Los Tensor Cores de tercera generación en la RTX 3050 proporcionan aceleración específica para cargas de trabajo de AI.</p>
                <p><strong>💰 Relación precio/rendimiento:</strong> Aunque la RTX 3050 cuesta ~67% más, ofrece {:.1f}x mejor rendimiento, justificando la inversión para aplicaciones AI.</p>
                <p><strong>⚡ Recomendación:</strong> Para segmentación semántica en producción, la RTX 3050 es la opción mínima recomendada para workflows eficientes.</p>
            </div>
        </div>
    </div>
    """.format(speed_improvement, fps_rtx/fps_gtx),
    unsafe_allow_html=True,
)

# Cerrar contenedor principal
st.markdown('</div>', unsafe_allow_html=True)
# Footer
st.markdown(
    """
    <div class="footer">
        <p>📊 Análisis Comparativo de Rendimiento - SegFormer Benchmarks</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Columna 3: Mejora
with col3:
    st.markdown(
        """
        <div class="content-card">
            <div class="card-content" style="text-align: center; padding: 2rem;">
                <h3 style="color: var(--text-color); margin-bottom: 0.5rem;">Mejora</h3>
                <div style="font-size: 2rem; font-weight: bold; color: var(--primary-color);">{:.1f}%</div>
                <p style="color: var(--text-light);">RTX vs GTX</p>
            </div>
        </div>
        """.format(speed_improvement),
        unsafe_allow_html=True,
    )

# Columna 4: FPS RTX
with col4:
    fps_rtx = 1 / rtx_avg if rtx_avg != 0 else 0

    st.markdown(
        """
        <div class="content-card">
            <div class="card-content" style="text-align: center; padding: 2rem;">
                <h3 style="color: var(--text-color); margin-bottom: 0.5rem;">FPS RTX</h3>
                <div style="font-size: 2rem; font-weight: bold; color: var(--primary-color);">{:.1f}</div>
                <p style="color: var(--text-light);">Frames por segundo</p>
            </div>
        </div>
        """.format(fps_rtx),
        unsafe_allow_html=True,
    )
