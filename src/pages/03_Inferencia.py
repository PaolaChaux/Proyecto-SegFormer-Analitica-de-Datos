import streamlit as st
from PIL import Image
import cv2
import tempfile
import os
import numpy as np

from segformer_model import (
    load_model, get_classes, segment_image, segment_frame,
    colorize_mask, mostrar_segmentacion_con_leyenda
)

st.set_page_config(page_title="3. Inferencia", page_icon="🤖")

@st.cache_resource
def get_model():
    return load_model()

# Cargar modelo
processor, model, device = get_model()
classes = get_classes(model)

st.title("Página 3: Inferencia en tiempo real desde webcam o archivo")

# Opción de webcam
start_cam = st.checkbox("Iniciar cámara en vivo")
stop_btn = st.button("Detener cámara", key="detener_cam_unique")

if start_cam and not stop_btn:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("No se pudo abrir la cámara. Verifica la conexión o los permisos.")
        st.stop()

    frame_slot = st.empty()
    info_slot = st.empty()
    frame_count = 0
    st.session_state["stop_cam"] = False

    while cap.isOpened() and not st.session_state.get("stop_cam", False):
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Procesar cada frame
        mask, duration = segment_frame(frame, processor, model, device)
        mask_color = colorize_mask(mask, classes)
        mask_resized = cv2.resize(mask_color, (frame.shape[1], frame.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
        overlay = cv2.addWeighted(frame, 0.5, mask_resized, 0.5, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        frame_slot.image(overlay_rgb, channels="RGB", 
                       caption=f"Inferencia en tiempo real: {duration:.3f}s")

        detected_idx = np.unique(mask)
        class_list = [classes[i] for i in detected_idx]
        info_slot.markdown("**Clases detectadas:** " + ", ".join(class_list))

    cap.release()
    st.write("Cámara detenida.")

if stop_btn:
    st.session_state["stop_cam"] = True

# Carga de archivo si no se usa webcam
if not start_cam:
    file = st.file_uploader("Sube una imagen (JPG/PNG) o video (MP4)", type=["jpg", "png", "mp4"])
    if file:
        if file.type.startswith("image"):
            image = Image.open(file).convert("RGB")
            st.image(image, caption="Imagen cargada")

            # Inferencia y comparativa solo para archivo
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

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            progress_bar = st.progress(0)
            frame_num = 0
            clases_detectadas_video = set()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                mask, _ = segment_frame(frame, processor, model, device)
                clases_frame = set(np.unique(mask))
                clases_detectadas_video.update(clases_frame)

                mask_color = colorize_mask(mask, classes)
                mask_resized = cv2.resize(mask_color, (frame.shape[1], frame.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)

                overlay = cv2.addWeighted(frame, 0.5, mask_resized, 0.5, 0)
                out.write(overlay)

                if frame_num == 0:
                    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), 
                           caption="Primer frame segmentado")

                frame_num += 1
                progress_bar.progress(min(frame_num / frame_count, 1.0))

            cap.release()
            out.release()

            st.success("Video segmentado procesado!")

            clases_detectadas_video = sorted(clases_detectadas_video)
            st.markdown("### Clases detectadas en el video:")
            for idx in clases_detectadas_video:
                st.markdown(f"- {classes[idx]}")

            st.video(output_path)
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Descargar video segmentado",
                    data=f,
                    file_name="video_segmentado.mp4",
                    mime="video/mp4"
                )