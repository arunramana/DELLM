# DELLM Minimal Implementation

Minimal working prototype following the minimal plan.

## Architecture

```
User Query
    ↓
[1. Query Decomposer] - Rule-based keyword matching
    ↓
[2. Cluster Router] - Routes by latency + fitness
    ↓
[3. Node Processing] - Parallel execution
    ↓
[4. Streaming Assembly] - Weighted voting
    ↓
[5. Transformer Synthesis] - Coherent answer
    ↓
[6. RL Training] - Fitness updates
    ↓
Final Answer
```

## Components

1. **decomposer.py** - Rule-based query decomposition
2. **router.py** - Routes chunks to nodes (0.7×latency + 0.3×(1-fitness))
3. **assembly.py** - Collects responses with weighted voting
4. **synthesis.py** - Synthesizes chunk results
5. **training.py** - Updates node fitness with rewards
6. **orchestrator.py** - Coordinates entire flow
7. **main.py** - FastAPI server entry point
8. **client.py** - Simple test client

## Usage

### Start Server

```bash
python minimal/main.py
```

### Test with Client

```bash
python minimal/client.py "Calculate 10% of 1000 then find best savings accounts"
```

### Or use API

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 2+2?"}'
```

## Key Features

- **Rule-based decomposition** - No LLM needed for splitting
- **Fitness-based routing** - Nodes selected by performance
- **Weighted voting** - Consensus with fitness × confidence
- **RL training** - Fitness updates after each query
- **Minimal code** - Simple, focused implementation

## Differences from Full Implementation

- No SuperLLM/SuperCluster hierarchy
- Rule-based decomposition (not LLM-based)
- Simpler routing formula
- Direct fitness updates (no complex verification)
- Single orchestrator (no multi-level coordination)

