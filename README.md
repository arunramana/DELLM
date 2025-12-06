# DELLM - Distributed Embedding-based LLM System

Minimal working prototype of a distributed LLM system using embedding-based processing with HuggingFace `transformers`.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Models

Edit `config/topology.json` to configure HuggingFace model names. The system uses:
- **Transformer Nodes**: Process embeddings through transformer blocks only (e.g., TinyLlama 1.1B)
- **Embedding Service**: Converts text to embeddings (server-side)
- **Decoder Service**: Converts processed embeddings back to text (server-side)

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
  -d '{"query": "what\'s 10% of 1000 and the tallest mountain"}'
```

## Architecture

```
User Query
    ↓
[1. Query Decomposer] - Rule-based splitting (preserves semantic meaning)
    ↓
[2. Embedding Service] - Converts text chunks to embeddings
    ↓
[3. Cluster Router] - Distributes chunks to all available nodes
    ↓
[4. Transformer Nodes] - Process embeddings through transformer blocks only
    ↓
[5. Embedding Assembly] - Collects processed embeddings with weighted voting
    ↓
[6. Embedding Assembler] - Assembles embeddings in order
    ↓
[7. Decoder Service] - Converts embeddings to text (per chunk)
    ↓
[8. Answer Combination] - Combines chunk answers in order
    ↓
[9. RL Training] - Updates node fitness based on performance
    ↓
Final Answer
```

## Key Features

- **Embedding-Based Processing**: Nodes process embeddings, not text
- **Semantic Decomposition**: Splits queries while preserving meaning
- **Parallel Processing**: Uses all available nodes efficiently
- **Transformer Blocks Only**: Nodes use only transformer layers (no embedding/decoding)
- **Web Search Integration**: Automatic web search for factual queries
- **Math Calculation**: Automatic calculation for percentage questions
- **Weighted Aggregation**: Combines results based on node fitness and confidence
- **Local Models**: Uses HuggingFace transformers (no external API calls)
- **Thread-Safe**: Handles concurrent model access safely

## Project Structure

```
DELLM/
├── minimal/                    # Core implementation
│   ├── decomposer.py          # Query decomposition
│   ├── router.py              # Node routing
│   ├── embedding_assembly.py  # Embedding collection & voting
│   ├── embedding_orchestrator.py  # Main coordinator
│   ├── training.py           # Fitness updates
│   ├── main.py               # FastAPI server
│   └── client.py             # Test client
├── core/
│   └── transformer_node.py   # Transformer node (processes embeddings)
├── utils/
│   ├── embedding_service.py  # Text → embeddings
│   ├── decoder_service.py    # Embeddings → text
│   ├── embedding_assembler.py  # Assembles embeddings
│   └── web_search.py         # Web search for factual queries
├── config/
│   └── topology.json         # Network topology & model names
├── tests/                     # Test files
│   ├── test_minimal.py
│   ├── test_decomposer.py
│   └── test_max_parallelism.py
├── check_server.py           # Server health check utility
├── requirements.txt
└── README.md
```

## Configuration

Edit `config/topology.json` to configure:
- Number of nodes
- Model name for each node (HuggingFace identifier, e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
- Embedding/decoder model names
- Initial fitness scores

Current topology: 5 transformer nodes

## How It Works

1. **Query Decomposition**: Splits query into semantic chunks (e.g., "what's 10% of 1000" and "tallest mountain")
2. **Embedding Generation**: Server converts each chunk to embeddings using embedding service
3. **Node Processing**: Transformer nodes process embeddings through transformer blocks only
4. **Embedding Assembly**: Processed embeddings are collected and assembled in order
5. **Decoding**: Each chunk's embeddings are decoded to text separately
6. **Answer Combination**: Answers are combined in order (e.g., "100 and Mount Everest")
7. **Web Search**: Factual queries automatically trigger web search for accurate information
8. **Math Calculation**: Percentage questions are automatically calculated correctly

## Testing

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

**Run tests:**
```bash
# From project root
python tests/test_minimal.py
python tests/test_decomposer.py
python tests/test_max_parallelism.py
```

## Implementation Details

- **Decomposition**: Rule-based splitting on natural separators ("then", "and", etc.). Preserves semantic units.
- **Routing**: Distributes chunks across all nodes. If more nodes than chunks, multiple nodes process same chunk in parallel.
- **Node Processing**: Each node processes embeddings through transformer blocks only (no embedding/decoding layers).
- **Embedding Assembly**: Weighted voting based on node fitness and embedding similarity.
- **Decoding**: Each chunk decoded separately to maintain order and accuracy.
- **Web Search**: Automatic for factual queries (tallest, highest, located in, etc.).
- **Math Calculation**: Automatic calculation for percentage questions.
- **Fitness Updates**: Simple RL - updates node fitness based on correctness and latency.

## Requirements

- Python 3.8+
- `transformers` (HuggingFace)
- `torch` (PyTorch)
- `accelerate` (for model loading)
- `beautifulsoup4` (for web search)
- FastAPI (for API server)

See `requirements.txt` for full dependency list.

## Notes

- Models are loaded once and reused (thread-safe)
- Node timeout: 120s (configurable in `minimal/embedding_assembly.py`)
- Client timeout: 300s (configurable in `minimal/client.py`)
- Models are automatically cleaned up on server shutdown
- Web search is enabled by default (can be disabled in orchestrator init)
