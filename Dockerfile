# 1) Imagen base con uv y Python 3.12
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 2) Directorio de trabajo
WORKDIR /app

# 3) Copiar solo lo que necesitas (data/docs están en .dockerignore)
COPY . .

# 4) Bytecode optimizado
ENV UV_COMPILE_BYTECODE=1

# 5) Actualizar pip y usar uv para instalar
RUN pip install --upgrade pip
RUN uv run pip install \
      --extra-index-url https://download.pytorch.org/whl/cu118 \
      -r requirements.txt

# 6) Exponer el puerto que Cloud Run usará
EXPOSE 8080

# 7) Comando para arrancar Streamlit apuntando al script correcto
CMD streamlit run src/appStreamlit.py \
    --server.port $PORT \
    --server.address 0.0.0.0

