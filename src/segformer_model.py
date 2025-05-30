import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

def load_model():
    # Determinar el dispositivo disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar el procesador y el modelo preentrenado
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")

    # Mover el modelo al dispositivo adecuado
    model.to(device)

    return processor, model, device

# Obtener clases del modelo
def get_classes(model):
    return [model.config.id2label[i] for i in range(len(model.config.id2label))]

# Segmentar imagen (PIL)
def segment_image(image, processor, model, device):
    # Procesar imagen y mover al dispositivo
    inputs = processor(images=image, return_tensors="pt").to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = logits.argmax(dim=1)[0]
    end = time.time()

    return prediction.cpu().numpy(), end - start

# Segmentar frame (cv2 image)
def segment_frame(frame, processor, model, device):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return segment_image(img, processor, model, device)

# Colorear máscara segmentada
def colorize_mask(mask, classes):
    cmap = plt.get_cmap("tab20", len(classes))
    mask_color = cmap(mask / np.max(mask))[:, :, :3]
    mask_color = (mask_color * 255).astype(np.uint8)
    return mask_color

# Mostrar segmentación con leyenda
def mostrar_segmentacion_con_leyenda(image, mask, classes, st):
    num_classes = len(classes)
    cmap = plt.get_cmap("tab20", num_classes)
    clases_detectadas_idx = np.unique(mask)
    clases_detectadas = [classes[i] for i in clases_detectadas_idx]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image)
    axs[0].set_title("Imagen Original")
    axs[0].axis('off')

    im = axs[1].imshow(mask, cmap=cmap, vmin=0, vmax=num_classes-1)
    axs[1].set_title("Segmentación")
    axs[1].axis('off')

    cbar = fig.colorbar(im, ax=axs[1], ticks=clases_detectadas_idx)
    cbar.ax.set_yticklabels(clases_detectadas)
    plt.tight_layout()
    st.pyplot(fig)
