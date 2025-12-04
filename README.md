# DELLM - Distributed LLM System

Minimal working prototype of a distributed LLM system using local models via `llama-cpp-python`.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Models

Edit `config/topology.json` to point to your local model files. The system uses:
- **Nodes**: Lightweight models (e.g., TinyLlama 1.1B) for parallel processing
- **Synthesis**: Larger model (e.g., Llama 3.2 3B) for answer synthesis

### 3. Start Server

```bash
python minimal/main.py
```

### 4. Test Query

```bash
python minimal/client.py
```

Or use the API directly:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Calculate 10% of 1000 then find best savings accounts"}'
```

## Architecture

```
User Query
    ↓
[1. Query Decomposer] - Rule-based splitting (preserves semantic meaning)
    ↓
[2. Cluster Router] - Distributes chunks to all available nodes
    ↓
[3. Node Processing] - Parallel execution with dual-model voting
    ↓
[4. Streaming Assembly] - Collects responses with weighted voting
    ↓
[5. Transformer Synthesis] - Combines results into coherent answer
    ↓
[6. RL Training] - Updates node fitness based on performance
    ↓
Final Answer
```

## Key Features

- **Semantic Decomposition**: Splits queries while preserving meaning (no meaningless fragments)
- **Parallel Processing**: Uses all available nodes efficiently
- **Dual-Model Voting**: Each node uses two models for consensus
- **Weighted Aggregation**: Combines results based on node fitness and confidence
- **Local LLMs**: Runs entirely on local models (no API calls)
- **Thread-Safe**: Handles concurrent model access safely

## Project Structure

```
DELLM/
├── minimal/              # Core implementation
│   ├── decomposer.py    # Query decomposition
│   ├── router.py        # Node routing
│   ├── assembly.py      # Response collection
│   ├── synthesis.py    # Answer synthesis
│   ├── training.py     # Fitness updates
│   ├── orchestrator.py # Main coordinator
│   ├── main.py         # FastAPI server
│   └── client.py       # Test client
├── core/
│   └── node.py         # Node execution (dual-model voting)
├── utils/
│   └── llm_client.py   # LLM wrapper (llama-cpp-python)
├── config/
│   └── topology.json   # Network topology & model paths
├── models/              # Local model files (GGUF format)
├── requirements.txt
└── README.md
```

## Configuration

Edit `config/topology.json` to configure:
- Number of nodes
- Model paths for each node (model_a_path, model_b_path)
- Synthesis model path (superllm.model_path)
- Initial fitness scores

Current topology: 5 nodes (4 in cluster, 1 direct) → 1 supercluster → 1 superllm

## Testing

**Component test:**
```bash
python test_minimal.py
```

**Decomposition test:**
```bash
python test_decomposer.py
```

**Parallelism test:**
```bash
python test_max_parallelism.py
```

**Full system:**
```bash
# Terminal 1
python minimal/main.py

# Terminal 2
python minimal/client.py
```

**Check server status:**
```bash
python check_server.py
```

## Implementation Details

- **Decomposition**: Rule-based splitting on natural separators ("then", "and", etc.). Preserves semantic units (won't split short phrases).
- **Routing**: Distributes chunks across all nodes. If more nodes than chunks, multiple nodes process same chunk in parallel.
- **Node Execution**: Each node runs two models (A and B), votes between them, returns consensus answer.
- **Synthesis**: Removes duplicates, takes best answer per chunk, combines into coherent final answer.
- **Fitness Updates**: Simple RL - updates node fitness based on correctness and latency.

## Requirements

- Python 3.8+
- `llama-cpp-python` (for local LLM inference)
- FastAPI (for API server)
- Local GGUF model files

See `requirements.txt` for full dependency list.

## Notes

- Models are loaded once and reused (thread-safe)
- Node timeout: 120s (configurable in `minimal/assembly.py`)
- Client timeout: 300s (configurable in `minimal/client.py`)
- Models are automatically cleaned up on server shutdown
