# DELLM: Minimal Implementation Plan
## Query to Answer Flow

---

## Architecture Overview

```
User Query
    ↓
[1. Query Decomposer] - Breaks query into vector chunks
    ↓
[2. Cluster Router] - Routes chunks to nodes (latency + fitness)
    ↓
[3. Node Processing] - Nodes process chunks in parallel, stream back
    ↓
[4. Streaming Assembly] - Collect node responses with consensus
    ↓
[5. Transformer Synthesis] - Small transformer creates coherent answer
    ↓
[6. RL Training] - Send feedback to nodes for learning
    ↓
Final Answer
```

---

## Component Details

### **1. Query Decomposer**
**Input**: User query string
**Output**: List of vector chunks

```python
chunks = [
    {
        'chunk_id': 'step_1',
        'operation': 'MATH_OP',
        'tokens': [100, 200, 300],
        'complexity': 0.4
    },
    ...
]
```

**Implementation**: Rule-based keyword matching (calculate, find, compare, generate)

---

### **2. Cluster Router** 
**Input**: Vector chunks
**Output**: Node assignments

```python
assignments = {
    'step_1': [node_1, node_2],  # 2 nodes for redundancy
    'step_2': [node_3],
}
```

**Routing Formula**:
```
score = 0.7 × latency + 0.3 × (1 - fitness)
Select node with minimum score
```

**Implementation**: Simple scoring + heap selection

---

### **3. Node Processing**
**Input**: Vector chunk + operation type
**Output**: Stream of tokens

```python
# Each node runs small LLM (1-3B params)
response = node.llm.generate(
    prompt=f"Operation: {operation}\nTokens: {tokens}\nAnswer:",
    stream=True
)
```

**Models**: 
- Phi-3-mini (3.8B)
- Qwen2.5-1.5B
- Llama-3.2-1B

**Implementation**: Use `transformers` library with streaming

---

### **4. Streaming Assembly**
**Input**: Multiple node responses per chunk
**Output**: Aggregated results

```python
# Weighted voting
for chunk_id, responses in node_responses:
    weights = [r.fitness * r.confidence for r in responses]
    winner = responses[argmax(weights)]
    aggregated[chunk_id] = winner.text
```

**Implementation**: Collect async streams, vote by fitness

---

### **5. Transformer Synthesis**
**Input**: Concatenated chunk results
**Output**: Coherent final answer

```python
combined = " | ".join([r.text for r in aggregated.values()])
prompt = f"Synthesize coherent answer: {combined}"
answer = small_transformer.generate(prompt)
```

**Model**: GPT-2 (1.5B) or Phi-2 (2.7B) on server

**Implementation**: Single forward pass, no streaming needed

---

### **6. RL Training**
**Input**: Final answer + verification
**Output**: Training signals to nodes

```python
# Calculate rewards
for node in nodes_used:
    if node.answer == correct_answer:
        reward = +0.05 + speed_bonus
    else:
        reward = -0.05
    
    # Update node fitness
    node.fitness = 0.9 * node.fitness + 0.1 * reward
    
    # Send feedback
    cluster.send_to_node(node_id, {
        'reward': reward,
        'correct_answer': correct_answer
    })
```

**Implementation**: 
- Fitness: Moving average (no model training initially)
- Later: Fine-tune with LoRA on (task, correct_answer) pairs

---

## Minimal Tech Stack

### **Decomposer**
- Python + regex/spaCy
- No models needed

### **Cluster** 
- FastAPI server
- Redis for node registry
- Python for routing logic

### **Nodes**
- `transformers` library
- Small LLMs (1-3B params)
- WebSocket/gRPC for streaming

### **Assembly**
- Async Python (`asyncio`)
- In-memory aggregation

### **Synthesis**
- `transformers` with GPT-2/Phi-2
- Single server GPU

### **Training**
- Fitness updates: In-memory
- LoRA fine-tuning: `peft` library (optional)

---

## Data Flow Example

**Query**: "Calculate 10% interest on $1000 then find best savings accounts"

### Step 1: Decompose
```
chunk_1: MATH_OP - "calculate 10% of 1000"
chunk_2: SEARCH_OP - "find best savings accounts"
```

### Step 2: Route
```
chunk_1 → [node_math_1, node_math_2] (redundancy)
chunk_2 → [node_search_1]
```

### Step 3: Process (Parallel)
```
node_math_1: "$100" (confidence: 0.95, latency: 0.8s)
node_math_2: "$100" (confidence: 0.92, latency: 1.2s)
node_search_1: "Found 5 accounts with 4.5% APY" (confidence: 0.88)
```

### Step 4: Aggregate
```
chunk_1 result: "$100" (unanimous consensus, confidence: 0.93)
chunk_2 result: "Found 5 accounts with 4.5% APY" (confidence: 0.88)
```

### Step 5: Synthesize
```
Input to transformer: "$100 | Found 5 accounts with 4.5% APY"
Output: "The interest on $1000 at 10% is $100. Based on this, 
         I found 5 savings accounts offering 4.5% APY."
```

### Step 6: Train
```
Verify answer correctness: ✅ (math correct, search relevant)

node_math_1: reward +0.06 (correct + fast)
node_math_2: reward +0.05 (correct + slower)
node_search_1: reward +0.04 (correct search)

Update fitness scores → Higher scores get more future tasks
```

---

## Implementation Timeline

### **Week 1: Core Infrastructure**
- Query decomposer (rule-based)
- Cluster routing logic
- Node simulator (mock responses)

### **Week 2: Real LLMs**
- Integrate small LLMs on nodes
- Streaming response collection
- Weighted voting assembly

### **Week 3: Synthesis**
- Add transformer synthesis layer
- End-to-end query flow
- Basic verification

### **Week 4: RL Training**
- Fitness scoring system
- Reward calculation
- Feedback loop to nodes

---

## Key Design Decisions

### **Why Redundancy?**
2 nodes per chunk → Consensus validation → Higher confidence

### **Why Small Models?**
1-3B params → Runs on consumer hardware (phones, laptops)

### **Why Streaming?**
Real-time token generation → Lower perceived latency

### **Why Synthesis Model?**
Nodes give fragments → Transformer creates coherence

### **Why RL?**
Nodes improve over time → System gets smarter without retraining

---

## Success Metrics

1. **Latency**: < 5s total (competitive with GPT-4)
2. **Accuracy**: > 85% correct answers
3. **Consensus**: > 70% unanimous agreement
4. **Fitness Growth**: +20% improvement over 1000 queries
5. **Scalability**: Support 100+ nodes

---

## Critical Path

1. ✅ Decomposer working
2. ✅ Cluster routing functional  
3. ✅ Streaming assembly proven
4. ⚠️ **Next**: Integrate real LLMs on nodes
5. ⚠️ **Next**: Add synthesis layer
6. ⚠️ **Next**: Close RL feedback loop

---

## MVP Scope (Simplifications)

- **No evolution**: Fixed node topology initially
- **No specialization**: All nodes general-purpose
- **No verification**: Trust consensus voting
- **No credits**: Just fitness scores
- **Single cluster**: No multi-cluster routing
- **Mock LLMs**: Use simulated responses first

Once MVP works → Add complexity incrementally.
