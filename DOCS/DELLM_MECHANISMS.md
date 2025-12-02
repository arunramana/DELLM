# DELLM Mechanisms

## Overview

This document describes the core mechanisms that enable DELLM's distributed, evolutionary architecture. These mechanisms work together to route queries, verify answers, and continuously evolve the network.

---

## 1. Query Simplicity Assessment

**Purpose:** Determine if a query needs decomposition or can be answered directly.

### Simplicity Score Calculation

```
Simplicity Score = weighted average of:
- Information needs (25%): How many facts required (1-5+)
- Logical steps (25%): Reasoning complexity (1-10+)
- Domain specificity (15%): How specialized (0-1)
- Ambiguity (15%): How clear the query is (0-1)
- Dependencies (10%): Cross-task dependencies (0-5+)
- Creativity (10%): Creative requirement (0-1)

Result: Score from 0.0 (very complex) to 1.0 (very simple)
```

### Routing Thresholds

- **Score ≥ 0.70**: Simple → Super LLM answers directly (no decomposition)
- **0.40-0.69**: Medium → Decompose and route to Clusters
- **< 0.40**: Complex → Must decompose into multiple subtasks

---

## 2. Problem Decomposition

**Location:** Super LLM Block

**Process:**
1. If simplicity score ≥ 0.70 → Super LLM answers directly (reduces latency)
2. If score < 0.70 → Decompose into fundamental sub-queries
3. Each sub-query gets its own Query Block with simplicity score
4. Group related queries together
5. Send each group to Super Cluster for parallel processing

**Output:** Multiple Query Blocks, each with:
- Sub-query text
- Simplicity score
- Classification (type, domain)
- Routing hint (node vs cluster)

---

## 3. Answer Aggregation

### 3.1 Super LLM Aggregation

**Process:**
1. Collect answers from all Super Clusters
2. Synthesize coherent final response
3. Resolve any conflicting information
4. Maintain logical consistency
5. Return to user

### 3.2 Verifier Aggregation

**Process:**
1. Verifier receives answers from nodes/clusters
2. Uses lightweight validation logic (NOT an LLM - just algorithms)
3. Assigns fitness score based on accuracy and latency
4. If fitness is high → aggregate and send up hierarchy
5. If fitness is low → re-route to best performing node/cluster

---

## 4. Verification

**Location:** Verifier Block (validation algorithms, NOT an LLM)

### Fitness Score Calculation

```
fitness = (accuracy × speed) / (latency + compute_cost)

Where:
- accuracy: correctness score (0-1)
- speed: tasks completed per hour
- latency: response time in seconds
- compute_cost: normalized resource usage
```

### Verification Process

1. **Node/Cluster returns answer** → Verifier receives it
2. **Fitness evaluation**:
   - High fitness (>0.75) → Accept answer, aggregate, send to parent
   - Low fitness (<0.75) → Re-route query to highest fitness node/cluster
3. **Verification methods**:
   - Redundancy voting (3 nodes vote on answer)
   - Programmatic validation (execute code, check math)
   - Consensus checking (flag conflicts)

### Fitness Thresholds

- **> 0.75**: Normal operation, answer accepted
- **0.55-0.75**: Warning zone, trigger light evolution
- **< 0.55**: Crisis mode, trigger aggressive evolution

---

## 5. Evolution

### 5.1 Topological Evolution (FREQUENT)

**What evolves:** Connections between Super Clusters ↔ Clusters ↔ Nodes

**Trigger:** Low fitness at cluster/super cluster level

**Process:**
1. Super Cluster/Cluster monitors fitness scores
2. If low fitness → explore new connections
3. Consider latency (CRITICAL) and fitness when making connections
4. Prune underperforming connections
5. Establish new connections to better nodes/clusters

**Connection Types:**
- Super LLM → Super Cluster
- Super Cluster → Cluster
- Super Cluster → Node (direct routing)
- Cluster → Node

**Criteria:** 
- Latency (primary - geographic proximity)
- Fitness score (secondary)
- Current load

### 5.2 Nodal Evolution (INFREQUENT)

**What evolves:** Node characteristics (NOT the LLM weights themselves)

**Node Chromosome:** Each node contains:
```
Chromosome = {
    model_type: str,           # "math_specialist" | "creative" | "code_specialist"
    confidence_threshold: float,
    temperature: float,
    top_p: float,
    top_k: int,
    prompt_template: str,
    specialization: str
}
```

**Process:**
1. Calculate fitness for all nodes in cluster
2. Bottom 20% → culled (killed)
3. Top 20% → spawn variants
4. Middle 60% → unchanged

**Note:** We DON'T modify LLM weights directly - only the parameters and configuration in the chromosome.

---

## 6. Mutation

**Trigger:** Cluster/Super Cluster detects low fitness (hostile environment)

**Target:** Only nodes can mutate (clusters/super clusters cannot)

**Process:**
1. Cluster/Super Cluster initiates mutation on connected node
2. Small random changes to chromosome:
   - Temperature ± 0.1
   - Top-p ± 0.05
   - Prompt template variations
   - Specialization drift

**Example:**
```
Before: {model_type: "math", temperature: 0.2, specialization: "algebra"}
After:  {model_type: "math", temperature: 0.3, specialization: "calculus"}
```

---

## 7. Crossover

**Trigger:** Cluster/Super Cluster in hostile environment (low fitness)

**Target:** Two high-performing nodes within the same cluster/super cluster

**Process:**
1. Select two high-fitness parent nodes
2. Combine their chromosomes
3. Create offspring with mixed traits
4. Add offspring to network

**Example:**
```
Parent A: {temperature: 0.2, model_type: "math", prompt: "detailed"}
Parent B: {temperature: 0.4, model_type: "fast", prompt: "concise"}

Offspring 1: {temperature: 0.3, model_type: "math", prompt: "concise"}
Offspring 2: {temperature: 0.3, model_type: "fast", prompt: "detailed"}
```

---

## 8. Culling

**Trigger:** Low fitness at cluster/super cluster level

**Process:**
1. Rank all connected nodes by fitness
2. Bottom 20% are killed (removed from network)
3. Freed resources used for spawning new nodes

**Frequency:** Every generation cycle (after M tasks or T time period)

---

## 9. Spawning

**Triggers:**
1. **New device joins network** → Spawn randomly initialized node
2. **After culling** → Spawn new nodes based on fitness
3. **Cluster/Super Cluster growth** → Spawn to meet demand

**Selection for spawning:**
- Probability proportional to fitness (fitter nodes more likely to spawn offspring)
- Top 20% performers spawn 2-3 variants each
- Variants have mutations from parent

**Initiated by:** Cluster or Super Cluster

---

## 10. Connections

### Connection Types

1. **Super LLM → Super Cluster**: Task routing
2. **Super Cluster → Cluster**: Group coordination
3. **Super Cluster → Node**: Direct routing (simple queries)
4. **Cluster → Node**: Task execution
5. **Node → Verifier**: Answer validation
6. **Cluster → Verifier**: Aggregated answer validation

### Connection Criteria

**Primary:** Latency (geographic proximity, network speed)
**Secondary:** Fitness score (historical performance)

**Process:**
1. Super Cluster/Cluster maintains connection pool
2. Periodically evaluate connection performance
3. Prune slow/low-fitness connections
4. Establish new connections based on latency + fitness
5. Load balance across connections

---

## 11. Super Cluster & Cluster Nature

### What They Are

**NOT**: Separate hardware or LLMs
**ARE**: Software programs running on user devices

### Function

**Coordination layer:**
- Route information between nodes
- Manage connections
- Calculate fitness scores
- Initiate evolution (mutation, crossover, spawning, culling)
- Track performance matrices

**Their Fitness:**
- Aggregated from fitness of connected nodes
- Modified by verifier based on routing decisions
- Drives their own evolutionary behavior

### Deployment

- Run as background services on contributor devices
- Lightweight (minimal CPU/memory)
- Can run multiple on powerful devices
- Self-organize into hierarchy

---

## Mechanism Summary

| Mechanism | Frequency | Trigger | Initiator | Target |
|-----------|-----------|---------|-----------|--------|
| Simplicity Assessment | Every query | Query arrival | Super LLM | Query Block |
| Decomposition | As needed | Low simplicity | Super LLM | Query |
| Verification | Every answer | Answer ready | Verifier | Answer |
| Topological Evolution | High | Low fitness | SC/Cluster | Connections |
| Nodal Evolution | Low | Generation cycle | SC/Cluster | Nodes |
| Mutation | Medium | Hostile environment | SC/Cluster | Nodes |
| Crossover | Low | Hostile environment | SC/Cluster | Nodes |
| Culling | Low | Generation cycle | SC/Cluster | Nodes |
| Spawning | Variable | Need/fitness | SC/Cluster | New nodes |

**Key Principle:** Topological evolution (changing connections) happens MORE frequently than nodal evolution (changing node characteristics) to avoid breaking existing models.

---

## Flow Example

```
1. User query arrives
   ↓
2. Super LLM calculates simplicity score
   ↓
3a. If simple (≥0.70) → Super LLM answers directly → User
   ↓
3b. If complex (<0.70) → Decompose into sub-queries
   ↓
4. Super Cluster routes sub-queries to Clusters/Nodes (based on simplicity + performance matrix)
   ↓
5. Nodes execute queries (dual-model voting)
   ↓
6. Verifier checks answers → calculates fitness
   ↓
7a. High fitness → aggregate → send up hierarchy
   ↓
7b. Low fitness → re-route to better node/cluster
   ↓
8. Super LLM receives all answers → synthesizes final response
   ↓
9. Return to user
   ↓
10. Background: Update fitness scores → trigger evolution if needed
```

---

*DELLM Mechanisms: Simple routing, intelligent evolution, continuous improvement.*
