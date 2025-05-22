# Arquitectura de SegFormer

![Arquitectura de SegFormer](Proyecto-SegFormer-Analitica-de-Datos/images/data/images/ARQUITECTURAIMAGEN.png)


## 1. Encoder (Codificador)

- El **encoder** de SegFormer es un **Transformer jerárquico**, diseñado para generar características a múltiples escalas. Utiliza un proceso llamado *merging* de parches superpuestos para reducir la resolución de las características mientras aumenta la dimensionalidad. Esto es diferente a los Transformers tradicionales, que generan características de baja resolución.

- **Entrada**: Un *patch* de imagen (de tamaño 4x4, como se describe en la arquitectura de SegFormer), que se transforma en una secuencia de tokens para procesarla.

- **Salida**: Genera características jerárquicas con resoluciones de diferentes tamaños (`H/4`, `H/8`, `H/16`, `H/32`), lo que le permite capturar tanto detalles finos como características de mayor contexto.

## 2. MLP Decoder (Decodificador)

- A diferencia de otros modelos que usan decodificadores complejos, SegFormer utiliza un **MLP (Perceptrón Multicapa)** sencillo como decodificador. Este modelo toma las características generadas por el encoder (a diferentes resoluciones) y las fusiona para generar la máscara de segmentación.

- **Entrada**: Características multi-escala generadas por el encoder.

- **Salida**: Una máscara de segmentación de resolución `H/4 x W/4` con un número de categorías determinado por el modelo (por ejemplo, 150 clases en ADE20K).

---

## Entradas del Modelo

- Las **entradas** del modelo son imágenes (en formato `PIL` o `cv2`) que se procesan mediante el `SegformerImageProcessor`. Este procesador convierte la imagen en tensores que el modelo puede manejar. La imagen es dividida en parches y convertida en tokens que son procesados por el **Transformer**.

  - **Función de entrada**:
    ```python
    processor(images=image, return_tensors="pt")
    ```
    donde `image` es la imagen que se desea segmentar.

---

## Salidas del Modelo

- Las **salidas** del modelo son las **máscaras de segmentación**. El modelo devuelve los **logits** (valores que indican la probabilidad de cada clase) para cada píxel en la imagen de entrada. Estos logits son procesados para obtener la clase de cada píxel usando `argmax`.

  - **Máscara de segmentación**: Imagen de salida donde cada píxel está etiquetado con una clase determinada.

  - **Tiempo de inferencia**: El tiempo que tarda en procesar la imagen, útil para evaluar la eficiencia del modelo.
