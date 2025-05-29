import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

st.set_page_config(
    page_title="Comparativa Inferencia", 
    page_icon="📊", 
    layout="wide")

st.title("Comparativa de tiempos de inferencia por computadora y video")

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

st.dataframe(df, use_container_width=True)

# Gráfico comparativo barras
fig, ax = plt.subplots(figsize=(2, 2))
indices = range(len(df))
width = 0.35

# Extraer tiempos totales como floats
def parse_time(s):
    return float(s.split('s')[0])

times_rtx = [parse_time(t) for t in df["RTX 3050, 4 GB VRAM"]]
times_gtx = [parse_time(t) for t in df["GTX 1650, 4 GB VRAM"]]

ax.bar(indices, times_rtx, width, label="RTX 3050")
ax.bar([i + width for i in indices], times_gtx, width, label="GTX 1650")

ax.set_xticks([i + width/2 for i in indices])
ax.set_xticklabels(df["Video"], rotation=30, ha="right")
ax.set_ylabel("Tiempo total (s)")
ax.set_title("Comparación de tiempo total de inferencia")
ax.legend()

buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches='tight')
buf.seek(0)

st.image(buf, width=700) 

