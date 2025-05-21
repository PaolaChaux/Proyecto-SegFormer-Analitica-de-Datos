# Usar una imagen base con Python 3.12 y uv preinstalado
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Habilitar la compilación de bytecode para mejorar el rendimiento
ENV UV_COMPILE_BYTECODE=1

# Copiar los archivos de configuración del proyecto
COPY pyproject.toml uv.lock ./

# Instalar las dependencias del proyecto sin incluir el código fuente
RUN uv sync --frozen --no-install-project --no-dev

# Copiar el código fuente del proyecto
COPY . .

# Instalar las dependencias del proyecto con el código fuente incluido
RUN uv sync --frozen --no-dev

# Añadir el entorno virtual al PATH
ENV PATH="/app/.venv/bin:$PATH"

# Establecer el punto de entrada para ejecutar el archivo Homepage.py
CMD ["uv", "run", "streamlit", "run", "src/appStreamlit.py.py", "--server.address=0.0.0.0", "--server.port=8080"]
