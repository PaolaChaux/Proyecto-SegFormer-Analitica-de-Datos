import streamlit as st
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np
import cv2
import tempfile
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="Segmentación Video SegFormer", layout="wide")

@st.cache_resource
def load_model():
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    return processor, model

processor, model = load_model()

classes = [
    'background', 'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road',
    'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door',
    'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting',
    'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat',
    'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion',
    'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand',
    'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
    'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway',
    'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower',
    'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island',
    'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus',
    'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight',
    'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole',
    'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster',
    'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer',
    'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall',
    'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
    'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher',
    'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light',
    'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board',
    'shower', 'radiator', 'glass', 'clock', 'flag'
]

def segment_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = logits.argmax(dim=1)[0].cpu().numpy()
    return pred

def colorize_mask(mask):
    cmap = plt.get_cmap("tab20", len(classes))
    mask_color = cmap(mask / np.max(mask))[:, :, :3]
    mask_color = (mask_color * 255).astype(np.uint8)
    return mask_color

st.title("Segmentación de video con SegFormer")

uploaded_file = st.file_uploader("Sube un video (MP4)", type=["mp4"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.write(f"Video cargado: {width}x{height} px, {fps:.2f} FPS")
    st.write(f"Número total de frames: {frame_count}")

    output_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress_bar = st.progress(0)
    frame_num = 0
    clases_detectadas_video = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pred_mask = segment_frame(frame)
        clases_frame = set(np.unique(pred_mask))
        clases_detectadas_video.update(clases_frame)

        mask_color = colorize_mask(pred_mask)
        mask_color_resized = cv2.resize(mask_color, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay = cv2.addWeighted(frame, 0.5, mask_color_resized, 0.5, 0)

        out.write(overlay)

        if frame_num == 0:
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Primer frame segmentado")

        frame_num += 1
        progress_bar.progress(min(frame_num / frame_count, 1.0))

    cap.release()
    out.release()

    st.success("Video segmentado procesado!")

    clases_detectadas_video = sorted(clases_detectadas_video)
    st.markdown("### Clases detectadas en el video:")
    for clase_idx in clases_detectadas_video:
        st.markdown(f"- {classes[clase_idx]}")

    st.video(output_path)

    with open(output_path, "rb") as f:
        st.download_button(
            label="Descargar video segmentado",
            data=f,
            file_name="video_segmentado.mp4",
            mime="video/mp4"
        )
