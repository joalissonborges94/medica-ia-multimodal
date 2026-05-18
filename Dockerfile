# =====================================================================
# Dockerfile do medica-ia-multimodal.
# Imagem base alinhada ao requires-python do pyproject.toml (>=3.12).
# Otimizado para Hugging Face Spaces (mesma imagem) e desenvolvimento local.
# =====================================================================

FROM python:3.12-slim AS base

# Variaveis de ambiente padrao para Python em container.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Dependencias de sistema necessarias para libs como opencv, librosa e ffmpeg.
# Mantido enxuto: cada nova lib pesada deve justificar entrada aqui.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia primeiro o requirements para aproveitar cache do Docker em builds incrementais.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia o restante do projeto.
COPY . .

# Pre-build do indice Chroma DURANTE o build da imagem, nao no boot.
# Em HF Spaces o boot do container tem timeout (~10min) e o build_rag_index
# leva ~5-8min (download bge-m3 + indexar 9 PDFs), fazendo o app nao
# responder em 7860 a tempo e disparando restart em loop.
# Movendo pra Docker RUN o app sobe em ~30s no boot (so carrega modelos
# em memoria; indice ja persiste no diretorio data/processed/chroma da
# imagem).
RUN python scripts/build_rag_index.py

EXPOSE 7860

# Entrypoint padrao: sobe o app Gradio diretamente. Indice Chroma ja foi
# construido durante o build da imagem.
CMD ["python", "app.py"]
