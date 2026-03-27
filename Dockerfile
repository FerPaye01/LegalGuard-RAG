# --- Etapa 1: Build & Dependencies ---
FROM python:3.10-slim AS builder

# Evitar que Python genere archivos .pyc y habilitar logs instantáneos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema necesarias para compilar algunas librerías de Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copiar requerimientos e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- HITO CRÍTICO: Descarga del Modelo spaCy en el Build ---
# Esto evita el timeout de 230s en Azure App Service
RUN python -m spacy download es_core_news_lg

# --- Etapa 2: Runtime ---
FROM builder AS runtime

WORKDIR /app

# Copiar el código fuente
COPY . .

# Crear directorios para outputs si no existen
RUN mkdir -p outputs/logs outputs/governance outputs/metrics data/temp

# Exponer el puerto por defecto de Streamlit
EXPOSE 8501

# Salud del contenedor (Healthcheck)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Comando de arranque optimizado para Azure App Service
ENTRYPOINT ["streamlit", "run", "src/frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
