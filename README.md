# DELLM Minimal Implementation

Minimal working prototype of distributed LLM system.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Server

```bash
python minimal/main.py
```

### 3. Test Query

```bash
python minimal/client.py "Calculate 10% of 1000 then find best savings accounts"
```

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

## Project Structure

```
dellm/
├── minimal/          # Minimal implementation
│   ├── decomposer.py
│   ├── router.py
│   ├── assembly.py
│   ├── synthesis.py
│   ├── training.py
│   ├── orchestrator.py
│   ├── main.py       # Server entry point
│   └── client.py     # Test client
├── core/
│   └── node.py       # Node execution
├── utils/
│   └── llm_client.py # LLM client
├── config/
│   └── topology.json # Network topology
├── models/           # Model files
└── requirements.txt
```

## Testing

**Component test:**
```bash
python test_minimal.py
```

**Full system:**
```bash
# Terminal 1
python minimal/main.py

# Terminal 2
python minimal/client.py "your query"
```

## Configuration

Edit `config/topology.json` to configure nodes and models.

See `DELLM_MINIMAL_PLAN.md` for detailed architecture.

