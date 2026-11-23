import os

bind = "0.0.0.0:8000"
# Each worker loads the embedding model (~0.5-1 GB RAM). Add workers per
# container or more containers (docker compose --scale app=N) to scale.
workers = int(os.environ.get("WEB_CONCURRENCY", 2))
threads = 1
timeout = 60
graceful_timeout = 30
keepalive = 5
# Recycle workers now and then so slow memory growth never takes a server down mid-exam
max_requests = 5000
max_requests_jitter = 500
accesslog = "-"
errorlog = "-"
