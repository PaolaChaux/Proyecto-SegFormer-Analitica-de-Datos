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

# Título principal
st.title("Página 3: Inferencia en tiempo real desde webcam o archivo")

# Cargar modelo con caché para evitar recarga
@st.cache_resource
def get_model():
    return load_model()

processor, model, device = get_model()
classes = get_classes(model)

# Sección cámara en vivo
start_cam = st.checkbox("Iniciar cámara en vivo")
stop_btn = st.button("Detener cámara", key="detener_cam_unique")

if start_cam and not stop_btn:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("No se pudo abrir la cámara. Verifica la conexión o los permisos.")
        st.stop()

    frame_slot = st.empty()
    info_slot = st.empty()
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

        frame_slot.image(overlay_rgb, channels="RGB", caption=f"Inferencia en tiempo real: {duration:.3f}s")

        detected_idx = np.unique(mask)
        class_list = [classes[i] for i in detected_idx]
        info_slot.markdown("**Clases detectadas:** " + ", ".join(class_list))

    cap.release()
    st.write("Cámara detenida.")

if stop_btn:
    st.session_state["stop_cam"] = True

# Sección carga archivo si no se usa cámara
if not start_cam:
    file = st.file_uploader("Sube una imagen (JPG/PNG) o video (MP4)", type=["jpg", "png", "mp4"])
    if file:
        if file.type.startswith("image"):
            image = Image.open(file).convert("RGB")
            st.image(image, caption="Imagen cargada")

            mask, duration = segment_image(image, processor, model, device)
            st.markdown(f"**Tiempo de inferencia:** {duration:.3f} segundos")
            mostrar_segmentacion_con_leyenda(image, mask, classes, st)

        elif file.type.endswith("mp4"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(file.read())
            video_path = tfile.name

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            st.write(f"Video cargado: {width}x{height} px, {fps:.2f} FPS")
            st.write(f"Número total de frames: {frame_count}")

            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "video_segmentado.mp4")

            progress_bar = st.progress(0)
            frame_num = 0
            clases_detectadas_video = set()

            tiempo_inicio_total = time.time()
            tiempo_total_inferencia = 0.0

            info_container = st.container()
            with info_container:
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

            st.success("✅ Video segmentado procesado!")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Tiempo Total", f"{tiempo_total_transcurrido:.2f}s", "Incluye I/O y procesamiento")
            with col2:
                st.metric("🧠 Tiempo de Inferencia", f"{tiempo_total_inferencia:.2f}s", "Tiempo solo inferencia")
            with col3:
                tiempo_promedio_frame = tiempo_total_inferencia / frame_count if frame_count > 0 else 0
                st.metric("📊 Promedio por Frame", f"{tiempo_promedio_frame:.3f}s", "Tiempo promedio inferencia por frame")

            st.markdown("### 📈 Estadísticas de Rendimiento")

            fps_procesamiento = frame_count / tiempo_total_transcurrido if tiempo_total_transcurrido > 0 else 0
            fps_inferencia = frame_count / tiempo_total_inferencia if tiempo_total_inferencia > 0 else 0

            col_stats1, col_stats2 = st.columns(2)
            with col_stats1:
                st.markdown(
                    f"""
                    **Velocidad de Procesamiento:**
                    - FPS video original: {fps:.2f}
                    - FPS procesamiento: {fps_procesamiento:.2f}
                    - FPS solo inferencia: {fps_inferencia:.2f}
                    """
                )
            with col_stats2:
                overhead_tiempo = tiempo_total_transcurrido - tiempo_total_inferencia
                porcentaje_inferencia = (tiempo_total_inferencia / tiempo_total_transcurrido * 100) if tiempo_total_transcurrido > 0 else 0

                st.markdown(
                    f"""
                    **Análisis de Tiempo:**
                    - Frames procesados: {frame_count:,}
                    - Overhead (I/O + escritura): {overhead_tiempo:.2f}s
                    - % tiempo en inferencia: {porcentaje_inferencia:.1f}%
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

            clases_detectadas_video = sorted(clases_detectadas_video)
            st.markdown("### 🏷️ Clases detectadas en el video:")

            num_cols = 3
            clases_por_columna = len(clases_detectadas_video) // num_cols + 1
            cols = st.columns(num_cols)

            for i, idx in enumerate(clases_detectadas_video):
                col_idx = i // clases_por_columna
                if col_idx < len(cols):
                    with cols[col_idx]:
                        st.markdown(f"• {classes[idx]}")

            st.markdown("### 🎬 Video Segmentado")
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button(
                    label="📥 Descargar video segmentado",
                    data=f,
                    file_name="video_segmentado.mp4",
                    mime="video/mp4"
                )
