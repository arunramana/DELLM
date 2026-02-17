import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "node/models/qwen2.5-1.5b-instruct-q8_0.gguf")
MODEL_N_CTX = 32000
MODEL_N_THREADS = int(os.getenv("MODEL_THREADS", "4"))
MODEL_TEMPERATURE = 0.2
MODEL_MAX_TOKENS = 2048

BOOTSTRAP_PEERS = [p for p in os.getenv("HIVEMIND_BOOTSTRAP_PEERS", "").split(",") if p]
DHT_PORT = int(os.getenv("DHT_PORT", "12346"))
WORKER_ID = os.getenv("WORKER_ID", f"worker_{os.getpid()}")

DOCKER_IMAGE = "python:3.11-slim"
DOCKER_TIMEOUT = 30

TASK_POLL_INTERVAL = 2
