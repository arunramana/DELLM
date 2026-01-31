# Recursive Coding Platform - MVP Setup

A peer-to-peer network of autonomous nodes that recursively decompose coding tasks, generate code, and test it distributedly.

## Quick Start

### Prerequisites
- Python 3.9+
- 16GB RAM recommended
- Docker installed and running

### Installation

```bash
# Clone repository
git clone <repo-url>
cd recursive-coding-platform

# Install dependencies
pip install hivemind llama-cpp-python docker asyncio pydantic

# Download model (one-time, ~1.5GB)
mkdir models
wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf -O models/qwen2.5-1.5b-instruct-q8_0.gguf
```

### Launch Network

```bash
# Start 6 nodes on your device
python network.py
```

### Submit Task

```bash
# In another terminal
python submit_task.py "Build a todo list component in React"
```

## Architecture

```
User Query → Entry Node → Decompose or Execute
                ↓
        ┌───────┼───────┐
        ↓       ↓       ↓
    Node A  Node B  Node C
    (delegate) (code) (code)
        ↓       ↓       ↓
    Integrate Results
        ↓
    Final Code
```

## How It Works

1. **Task Submission**: User sends coding task to any node
2. **Decision**: Node decides to execute directly or delegate
3. **Recursion**: Complex tasks decompose into subtasks
4. **Execution**: Atomic tasks generate code
5. **Testing**: Code runs in Docker sandbox
6. **Integration**: Parent nodes assemble child results
7. **Delivery**: Final code returned to user

## Node Capabilities

Each node can:
- ✅ Generate code using Qwen2.5-1.5B
- ✅ Decompose tasks into subtasks
- ✅ Delegate to peer nodes
- ✅ Test code in Docker sandbox
- ✅ Integrate results from children

## Configuration

Edit `config.py`:
```python
NUM_NODES = 6              # Number of nodes to launch
MODEL_PATH = "models/..."  # Path to GGUF model
MAX_RETRIES = 3           # Code generation retry limit
TASK_TIMEOUT = 60         # Seconds to wait for subtasks
```

## Expected Resource Usage

- 6 nodes × 2GB = 12GB RAM
- 1.5GB disk for model
- Minimal CPU (inference on-demand)

## Troubleshooting

**Nodes not discovering peers?**
- Check firewall settings
- Ensure all nodes have network access

**Code execution failing?**
- Verify Docker is running: `docker ps`
- Check container limits in node.py

**Out of memory?**
- Reduce NUM_NODES in config
- Close other applications

## What This MVP Demonstrates

✅ P2P node network with Hivemind DHT  
✅ Autonomous task delegation  
✅ Recursive problem decomposition  
✅ Distributed code generation  
✅ Sandboxed code execution  
✅ Result integration  

## Next Steps

After MVP works:
1. Add verification/voting between nodes
2. Implement contract enforcement
3. Add mission context propagation
4. Build web interface
5. Support multi-device networks

## Files

```
recursive-coding-platform/
├── network.py          # Launch multiple nodes
├── node.py            # Node implementation
├── submit_task.py     # Submit coding tasks
├── config.py          # Configuration
├── models/            # LLM weights
└── README.md
```

## License

MIT
