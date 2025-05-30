# SegFormer: Arquitectura Transformer para Segmentación Semántica

SegFormer es un modelo desarrollado por **NVIDIA** que combina la potencia de las arquitecturas Transformer con eficiencia y precisión para la tarea de segmentación semántica en imágenes.

## ¿Qué es SegFormer?

- Es un modelo basado en Transformers diseñado para segmentación semántica.
- Combina un encoder jerárquico sin embeddings posicionales con un decoder ligero.
- Permite segmentar imágenes con alta precisión y eficiencia computacional.

## Arquitectura

### Encoder: Mix Transformer (MiT)

- Inspirado en el Swin Transformer, cuenta con un diseño jerárquico que procesa las imágenes en múltiples resoluciones.
- Las escalas de procesamiento son: 1/4, 1/8, 1/16 y 1/32 de la resolución original.
- Mezcla mecanismos de atención local y global para capturar tanto detalles finos como contexto amplio.
- La versión más ligera, MiT-B0, cuenta con aproximadamente 3.7 millones de parámetros.

### Decoder: MLP Eficiente

- Un decoder simple basado en capas MLP que proyecta y fusiona características de distintas escalas.
- Su diseño ligero mantiene la eficiencia sin sacrificar precisión.
- Cuenta con alrededor de 0.4 millones de parámetros.

## Ventajas Clave

- **Sin Embeddings Posicionales:** Esto permite que el modelo trabaje con imágenes de tamaños variados sin perder precisión espacial.
- **Jerarquía eficiente:** El encoder jerárquico proporciona un equilibrio óptimo entre contexto global y detalles locales.
- **Alta Velocidad de Inferencia:** Capaz de procesar imágenes hasta a 47 FPS, ideal para aplicaciones en tiempo real.
- **Modularidad:** Dispone de diferentes versiones (MiT-B0 a MiT-B5) para adaptar el modelo a distintos requerimientos de recursos y precisión.

## Parámetros Totales (MiT-B0)

| Componente | Parámetros (Millones) |
|------------|----------------------|
| Encoder    | 3.7                  |
| Decoder    | 0.4                  |
| **Total**  | **~4.1**             |

## Referencias

- Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers", NeurIPS 2021  
  [📄 Paper en arXiv](https://arxiv.org/abs/2105.15203)
