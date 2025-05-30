# 03_Inferencia.py
import streamlit as st
from PIL import Image
import cv2
import tempfile
import os
import numpy as np
import time

from segformer_model import (
    load_model, get_classes, segment_image, segment_frame,
    colorize_mask, mostrar_segmentacion_con_leyenda
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
            <p class="hero-subtitle">Segmentación semántica desde webcam o archivos</p>
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

# Sección cámara en vivo
st.markdown(
    """
    <div class="demo-section">
        <div class="section-header">
            <h2>📹 Inferencia desde Webcam</h2>
            <p>Segmentación semántica en tiempo real usando tu cámara</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Controles de cámara en tarjeta
st.markdown(
    """
    <div class="content-card">
        <div class="card-header">
            <h2>Controles de Cámara</h2>
        </div>
        <div class="card-content">
    """,
    unsafe_allow_html=True,
)

left_space, col1, col2, right_space = st.columns([1, 2, 2, 1])

with col1:
    start_cam = st.button("🟢 Activar cámara", use_container_width=True)

with col2:
    stop_btn = st.button("🔴 Detener cámara", key="detener_cam_unique", use_container_width=True)

st.markdown('</div></div>', unsafe_allow_html=True)

if start_cam and not stop_btn:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.markdown(
            """
            <div class="warning-box">
                <div class="warning-icon">⚠️</div>
                <div class="warning-text">
                    <strong>Error de Cámara</strong><br>
                    No se pudo abrir la cámara. Verifica la conexión o los permisos.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    # Contenedor centrado para video e info
    st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; margin-top: 1rem;">
            <div style="max-width: 800px; width: 100%;">
    """, unsafe_allow_html=True)

    frame_slot = st.empty()
    info_slot = st.empty()

    st.markdown("""
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.session_state["stop_cam"] = False

    while cap.isOpened() and not st.session_state.get("stop_cam", False):
        ret, frame = cap.read()
        if not ret:
            break

        mask, duration = segment_frame(frame, processor, model, device)
        mask_color = colorize_mask(mask, classes)
        mask_resized = cv2.resize(mask_color, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = cv2.addWeighted(frame, 0.5, mask_resized, 0.5, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        frame_slot.image(overlay_rgb, channels="RGB", caption=f"⚡ Inferencia: {duration:.3f}s")

        detected_idx = np.unique(mask)
        class_list = [classes[i] for i in detected_idx]
        info_slot.markdown(
            f"""
            <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0; text-align: center;">
                <strong>🏷️ Clases detectadas:</strong> {", ".join(class_list)}
            </div>
            """, 
            unsafe_allow_html=True
        )

    cap.release()
    st.success("🟢 Cámara detenida correctamente.")

if stop_btn:
    st.session_state["stop_cam"] = True

# Sección carga archivo si no se usa cámara
if not start_cam:
    st.markdown(
        """
        <div class="demo-section">
            <div class="section-header">
                <h2>📁 Inferencia desde Archivo</h2>
                <p>Sube una imagen o video para procesamiento</p>
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
        type=["jpg", "png", "mp4"],
        help="Formatos soportados: JPG, PNG para imágenes | MP4 para videos"
    )
    
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    if file:
        if file.type.startswith("image"):
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


        elif file.type.endswith("mp4"):
            # Procesamiento de video
            st.markdown(
                """
                <div class="content-card">
                    <div class="card-header">
                        <h2>🎬 Procesando Video</h2>
                    </div>
                    <div class="card-content">
                """,
                unsafe_allow_html=True,
            )
            
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(file.read())
            video_path = tfile.name

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Información del video
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📐 Resolución", f"{width}x{height} px")
            with col2:
                st.metric("🎞️ FPS", f"{fps:.2f}")
            with col3:
                st.metric("📊 Total Frames", f"{frame_count:,}")

            st.markdown('</div></div>', unsafe_allow_html=True)

            # Procesamiento del video
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "video_segmentado.mp4")

            st.markdown(
                """
                <div class="content-card tech-card">
                    <div class="card-header">
                        <div class="tech-icon">⚙️</div>
                        <h2>Progreso de Procesamiento</h2>
                    </div>
                    <div class="card-content">
                """,
                unsafe_allow_html=True,
            )

            progress_bar = st.progress(0)
            frame_num = 0
            clases_detectadas_video = set()

            tiempo_inicio_total = time.time()
            tiempo_total_inferencia = 0.0

            st.info("🔄 Procesando video... Esto puede tomar unos minutos.")

            fourcc = cv2.VideoWriter_fourcc(*"h264")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                mask, tiempo_frame = segment_frame(frame, processor, model, device)
                tiempo_total_inferencia += tiempo_frame

                clases_frame = set(np.unique(mask))
                clases_detectadas_video.update(clases_frame)

                mask_color = colorize_mask(mask, classes)
                mask_resized = cv2.resize(mask_color, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                overlay = cv2.addWeighted(frame, 0.5, mask_resized, 0.5, 0)
                out.write(overlay)

                frame_num += 1
                progress_bar.progress(min(frame_num / frame_count, 1.0))

            cap.release()
            out.release()

            tiempo_fin_total = time.time()
            tiempo_total_transcurrido = tiempo_fin_total - tiempo_inicio_total

            st.markdown('</div></div>', unsafe_allow_html=True)

            # Resultados del procesamiento
            st.markdown(
                """
                <div class="content-card highlight">
                    <div class="card-header">
                        <h2>✅ Procesamiento Completado</h2>
                    </div>
                    <div class="card-content">
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Tiempo Total", f"{tiempo_total_transcurrido:.2f}s", "Incluye I/O y procesamiento")
            with col2:
                st.metric("🧠 Tiempo de Inferencia", f"{tiempo_total_inferencia:.2f}s", "Tiempo solo inferencia")
            with col3:
                tiempo_promedio_frame = tiempo_total_inferencia / frame_count if frame_count > 0 else 0
                st.metric("📊 Promedio por Frame", f"{tiempo_promedio_frame:.3f}s", "Tiempo promedio inferencia por frame")

            st.markdown('</div></div>', unsafe_allow_html=True)

            # Estadísticas de rendimiento
            st.markdown(
                """
                <div class="content-card">
                    <div class="card-header">
                        <h2>📈 Estadísticas de Rendimiento</h2>
                    </div>
                    <div class="card-content">
                """,
                unsafe_allow_html=True,
            )

            fps_procesamiento = frame_count / tiempo_total_transcurrido if tiempo_total_transcurrido > 0 else 0
            fps_inferencia = frame_count / tiempo_total_inferencia if tiempo_total_inferencia > 0 else 0

            col_stats1, col_stats2 = st.columns(2)
            with col_stats1:
                st.markdown(
                    f"""
                    **🚀 Velocidad de Procesamiento:**
                    - FPS video original: `{fps:.2f}`
                    - FPS procesamiento: `{fps_procesamiento:.2f}`
                    - FPS solo inferencia: `{fps_inferencia:.2f}`
                    """
                )
            with col_stats2:
                overhead_tiempo = tiempo_total_transcurrido - tiempo_total_inferencia
                porcentaje_inferencia = (tiempo_total_inferencia / tiempo_total_transcurrido * 100) if tiempo_total_transcurrido > 0 else 0

                st.markdown(
                    f"""
                    **⚡ Análisis de Tiempo:**
                    - Frames procesados: `{frame_count:,}`
                    - Overhead (I/O + escritura): `{overhead_tiempo:.2f}s`
                    - % tiempo en inferencia: `{porcentaje_inferencia:.1f}%`
                    """
                )

            file_size = os.path.getsize(output_path) / (1024 * 1024)
            st.markdown(
                f"""
                **📄 Información del Archivo:**
                - Códec: **H.264**
                - Formato: **MP4**
                - Tamaño: **{file_size:.2f} MB**
                """
            )

            st.markdown('</div></div>', unsafe_allow_html=True)

            # Clases detectadas
            st.markdown(
                """
                <div class="content-card tech-card">
                    <div class="card-header">
                        <div class="tech-icon">🏷️</div>
                        <h2>Clases Detectadas en el Video</h2>
                    </div>
                    <div class="card-content">
                """,
                unsafe_allow_html=True,
            )

            clases_detectadas_video = sorted(clases_detectadas_video)
            
            num_cols = 3
            clases_por_columna = len(clases_detectadas_video) // num_cols + 1
            cols = st.columns(num_cols)

            for i, idx in enumerate(clases_detectadas_video):
                col_idx = i // clases_por_columna
                if col_idx < len(cols):
                    with cols[col_idx]:
                        st.markdown(f"• `{classes[idx]}`")

            st.markdown('</div></div>', unsafe_allow_html=True)

            # Video resultado
            st.markdown(
                """
                <div class="carousel-container">
                    <h3 style="text-align: center; margin-bottom: 1rem;">🎬 Video Segmentado</h3>
                """,
                unsafe_allow_html=True,
            )
            
            st.video(output_path)
            
            st.markdown('</div>', unsafe_allow_html=True)

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