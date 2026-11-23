# Search website image: gunicorn + Flask app, embedding model baked in so
# containers start without internet access.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    # one or two CPU threads per worker; scale with more workers/replicas instead
    OMP_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Newer pip resumes interrupted downloads (still hash-checked); flaky networks
# otherwise cut the 170 MB torch wheel short and the build fails.
ENV PIP_NO_CACHE_DIR=1 PIP_TIMEOUT=60 PIP_RETRIES=10 PIP_RESUME_RETRIES=20
RUN pip install -U "pip>=25.1"
# CPU-only torch: the CUDA build is several GB and the server has no GPU
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

RUN useradd --create-home appuser \
 && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')" \
 && chown -R appuser /opt/hf
# Use the baked-in model; never reach out to the Hugging Face Hub at runtime
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY app.py common.py gunicorn.conf.py papers_catalog.jsonl ./
COPY templates ./templates

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
