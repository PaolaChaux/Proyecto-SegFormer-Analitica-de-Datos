from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import requests
import time
import matplotlib.pyplot as plt
import numpy as np

# Lista completa de clases ADE20k (150 clases)
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

# Cargar imagen desde URL
url = "https://sillaoficina365.es/img/cms/BLOG/JUNIO/03/imagen-2.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# Cargar modelo y procesador pretrained
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# Medir tiempos para preprocesamiento e inferencia
start_total = time.time()

start_pre = time.time()
inputs = processor(images=image, return_tensors="pt")
end_pre = time.time()

start_inf = time.time()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = logits.argmax(dim=1)[0]
end_inf = time.time()

end_total = time.time()

print(f"Tiempo preprocesamiento: {end_pre - start_pre:.3f} s")
print(f"Tiempo inferencia: {end_inf - start_inf:.3f} s")
print(f"Tiempo total: {end_total - start_total:.3f} s")

# Visualizar imagen original y segmentación
num_classes = len(classes)
cmap = plt.get_cmap("tab20", num_classes)

plt.figure(figsize=(14,7))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Imagen original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(prediction.cpu().numpy(), cmap=cmap, vmin=0, vmax=num_classes-1)
cbar = plt.colorbar(ticks=range(0, num_classes, 10), fraction=0.046, pad=0.04)
# Mostrar etiquetas simplificadas en el colorbar cada 10 clases para no saturar
labels_to_show = [classes[i] if i < len(classes) else '' for i in range(0, num_classes, 10)]
cbar.ax.set_yticklabels(labels_to_show)
plt.title("Segmentación Semántica")
plt.axis("off")

plt.show()
