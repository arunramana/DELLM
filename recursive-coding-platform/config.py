# Configuration for Recursive Coding Platform

# Number of nodes to launch on this device
NUM_NODES = 6

# Path to model weights
MODEL_PATH = "models/qwen2.5-1.5b-instruct-q8_0.gguf"

# LLM settings
N_CTX = 4096              # Context window size
N_THREADS = 4             # CPU threads per model
N_GPU_LAYERS = 0          # Set > 0 to use GPU

# Task settings
MAX_RETRIES = 3           # Code generation retry limit
TASK_TIMEOUT = 300        # Seconds to wait for subtask results (increased for complex tasks)
HEARTBEAT_INTERVAL = 20   # Seconds between DHT updates

# Code execution
DOCKER_IMAGE = "python:3.11-slim"
DOCKER_MEM_LIMIT = "256m"
DOCKER_TIMEOUT = 10       # Seconds

# DHT settings
DHT_PORT = 8000           # Starting port for DHT
EXPIRATION_TIME = 30      # Seconds before DHT entries expire
