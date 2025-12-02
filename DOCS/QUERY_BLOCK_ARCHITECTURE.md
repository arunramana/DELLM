# Query Block: Detailed Architecture Specification

## Overview

The Query Block is the fundamental data structure in DELLM that represents a question or task to be answered. It flows through the entire system hierarchy, from initial user input through Super LLM decomposition to individual node execution.

**Key Principle:** Query Blocks exist at every level of the hierarchy:
- **User Query Block** - Original question from the user
- **Decomposed Query Blocks** - Subtasks created by Super LLM
- **Simple Query Blocks** - Individual queries routed to nodes

**Routing Decision:** Each Query Block has a simplicity score that determines optimal routing:
- **Simple queries (score ≥ 0.70)** → Routed directly to Nodes
- **Medium queries (0.40-0.69)** → Routed to Clusters
- **Complex queries (< 0.40)** → Decomposed by Super LLM

---

## Data Structure

### Core Query Block

```python
QueryBlock = {
    # ========================================
    # IDENTIFICATION
    # ========================================
    query_id: str,           # "query-abc123"
    parent_id: str | None,   # Parent query if this is a subtask
    batch_id: str | None,    # Batch ID if part of decomposition
    depth: int,              # 0 = user query, 1+ = decomposed
    
    # ========================================
    # THE QUERY ITSELF
    # ========================================
    query: str,              # The actual question/task
    
    # ========================================
    # SIMPLICITY ASSESSMENT (CRITICAL)
    # ========================================
    simplicity_score: float, # 0.0 to 1.0
    # 0.00-0.39: Very complex (must decompose)
    # 0.40-0.54: Complex (should decompose)
    # 0.55-0.69: Medium (route to cluster)
    # 0.70-0.84: Simple (route to nodes)
    # 0.85-1.00: Very simple (route to nodes)
    
    simplicity_reasoning: str,  # Why this score was assigned
    
    # Simplicity Factors
    simplicity_factors: {
        information_needs: int,      # How many facts needed (1-5+)
        logical_steps: int,          # Reasoning steps required (1-10+)
        domain_specificity: float,   # How specialized (0-1)
        ambiguity: float,           # How clear the ask is (0-1, lower=clearer)
        dependencies: int,          # Cross-subtask dependencies (0-5+)
        creativity_required: float  # How much creativity (0-1)
    },
    
    # ========================================
    # CLASSIFICATION
    # ========================================
    classification: {
        # Primary type
        task_type: str,  # "factual" | "calculation" | "reasoning" | 
                         # "creative" | "coding" | "analysis" | "search"
        
        # Domain
        domain: str,     # "mathematics" | "science" | "creative_writing" |
                         # "programming" | "general" | "travel" | etc.
        
        # Complexity (from Super LLM or Cluster analysis)
        complexity: str, # "trivial" | "simple" | "medium" | "complex" | "very_complex"
        
        # Requirements
        requires_internet: bool,
        requires_calculation: bool,
        requires_code_execution: bool,
        requires_multimodal: bool
    },
    
    # ========================================
    # ROUTING INFORMATION
    # ========================================
    routing: {
        # Current routing decision
        current_route: str,  # "super_llm" | "super_cluster" | "cluster" | "node"
        
        # Super LLM's routing hint (if decomposed)
        super_llm_hint: {
            suggested_routing: str,  # "node" | "cluster" | "any"
            confidence: float,        # 0-1
            reasoning: str
        } | None,
        
        # Routing history (as query moves through system)
        routing_history: [
            {
                timestamp: float,
                from_block: str,      # "super-llm-main"
                to_block: str,        # "super-cluster-main"
                routing_decision: str,
                reasoning: str
            }
        ]
    },
    
    # ========================================
    # CONSTRAINTS
    # ========================================
    constraints: {
        max_latency_seconds: float,   # Time budget
        min_confidence: float,         # Minimum acceptable confidence
        priority: str,                # "low" | "medium" | "high" | "critical"
        
        # Resource limits
        max_tokens: int | None,
        max_iterations: int | None,
        
        # Quality requirements
        verification_required: bool,
        min_verifier_consensus: float
    },
    
    # ========================================
    # CONTEXT
    # ========================================
    context: {
        # User context
        user_id: str,
        session_id: str,
        conversation_history: [str] | None,  # Previous messages if relevant
        
        # Query context
        follow_up_to: str | None,    # Previous query_id if continuation
        requires_context_from: [str] | None,  # Other query_ids needed
        
        # Environmental context
        timestamp: float,
        user_location: str | None,
        user_preferences: dict | None
    },
    
    # ========================================
    # METADATA
    # ========================================
    metadata: {
        created_at: float,
        created_by: str,         # "user" | "super-llm-main"
        version: str,            # For tracking schema changes
        
        # Performance tracking
        expected_quality: float,
        expected_latency: float,
        
        # Debugging
        debug_mode: bool,
        trace_enabled: bool
    }
}
```

---

## Simplicity Score Calculation

### Algorithm

The simplicity score is the MOST CRITICAL field because it determines routing decisions throughout DELLM.

```python
def calculate_simplicity_score(query: str, classification: dict) -> dict:
    """
    Calculate simplicity score based on multiple factors.
    
    Returns:
    {
        simplicity_score: float (0-1),
        reasoning: str,
        factors: dict
    }
    """
    # Step 1: Analyze query using LLM (Super LLM or Cluster)
    analysis = analyze_query_complexity(query)
    
    # Step 2: Extract factors
    factors = {
        "information_needs": analysis.num_facts_needed,      # 1-5+
        "logical_steps": analysis.reasoning_steps,           # 1-10+
        "domain_specificity": analysis.specialization,       # 0-1
        "ambiguity": analysis.clarity_score,                # 0-1 (inverted)
        "dependencies": analysis.cross_dependencies,         # 0-5+
        "creativity_required": analysis.creativity_level    # 0-1
    }
    
    # Step 3: Calculate weighted score
    # More information needs → lower simplicity
    info_score = max(0, 1.0 - (factors["information_needs"] - 1) * 0.2)
    
    # More logical steps → lower simplicity
    logic_score = max(0, 1.0 - (factors["logical_steps"] - 1) * 0.1)
    
    # More specialized → lower simplicity
    domain_score = 1.0 - factors["domain_specificity"]
    
    # More ambiguous → lower simplicity
    clarity_score = 1.0 - factors["ambiguity"]
    
    # More dependencies → lower simplicity
    dependency_score = max(0, 1.0 - factors["dependencies"] * 0.2)
    
    # More creativity → lower simplicity
    creativity_score = 1.0 - factors["creativity_required"]
    
    # Weighted average
    simplicity_score = (
        info_score * 0.25 +
        logic_score * 0.25 +
        domain_score * 0.15 +
        clarity_score * 0.15 +
        dependency_score * 0.10 +
        creativity_score * 0.10
    )
    
    # Step 4: Generate reasoning
    reasoning = generate_simplicity_reasoning(factors, simplicity_score)
    
    return {
        "simplicity_score": round(simplicity_score, 2),
        "reasoning": reasoning,
        "factors": factors
    }
```

### Examples of Simplicity Scores

```python
# Very Simple (0.85-1.00) - Direct to Node
{
    "query": "What is 25 * 17?",
    "simplicity_score": 0.95,
    "factors": {
        "information_needs": 1,     # Just needs to calculate
        "logical_steps": 1,         # Single multiplication
        "domain_specificity": 0.0,  # Basic arithmetic
        "ambiguity": 0.0,          # Crystal clear
        "dependencies": 0,          # Standalone
        "creativity_required": 0.0  # None
    }
}

# Simple (0.70-0.84) - Direct to Node
{
    "query": "What is the capital of France?",
    "simplicity_score": 0.78,
    "factors": {
        "information_needs": 1,     # Single fact
        "logical_steps": 1,         # Direct lookup
        "domain_specificity": 0.1,  # Geography basics
        "ambiguity": 0.0,          # Very clear
        "dependencies": 0,          # Standalone
        "creativity_required": 0.0  # None
    }
}

# Medium (0.55-0.69) - Route to Cluster
{
    "query": "Find hotels in Tokyo under $150/night with good reviews",
    "simplicity_score": 0.62,
    "factors": {
        "information_needs": 3,     # Hotels, prices, reviews
        "logical_steps": 3,         # Search, filter, rank
        "domain_specificity": 0.3,  # Travel/hospitality
        "ambiguity": 0.2,          # "Good" is subjective
        "dependencies": 0,          # Standalone
        "creativity_required": 0.1  # Minimal
    }
}

# Complex (0.40-0.54) - Should Decompose
{
    "query": "Compare the economic policies of Roosevelt and Reagan",
    "simplicity_score": 0.48,
    "factors": {
        "information_needs": 4,     # Both policies, context, comparison
        "logical_steps": 5,         # Research, analyze, compare, synthesize
        "domain_specificity": 0.6,  # Political economy
        "ambiguity": 0.3,          # "Compare" needs clarification
        "dependencies": 1,          # Must understand both before comparing
        "creativity_required": 0.3  # Analysis required
    }
}

# Very Complex (< 0.40) - Must Decompose
{
    "query": "Plan a 2-week trip to Japan during cherry blossom season under $3000",
    "simplicity_score": 0.35,
    "factors": {
        "information_needs": 5,     # Flights, hotels, itinerary, costs, timing
        "logical_steps": 8,         # Search flights, calc budget, find hotels, plan days...
        "domain_specificity": 0.4,  # Travel planning
        "ambiguity": 0.4,          # Many decisions to make
        "dependencies": 3,          # Budget affects hotels, timing affects availability
        "creativity_required": 0.5  # Significant planning needed
    }
}
```

---

## Routing Logic Based on Simplicity

### Super Cluster Routing Decision

```python
class SuperClusterRouter:
    def route_query(self, query: QueryBlock) -> str:
        """
        Decide where to route query based on simplicity score.
        
        Returns: "node" | "cluster"
        """
        score = query.simplicity_score
        
        # Decision thresholds
        if score >= 0.70:
            # Simple enough for direct node execution
            decision = "node"
            reasoning = f"Simplicity {score:.2f} ≥ 0.70 threshold - route to node"
            
        elif score >= 0.40:
            # Medium complexity - benefit from cluster coordination
            decision = "cluster"
            reasoning = f"Simplicity {score:.2f} in [0.40, 0.70) - route to cluster"
            
        else:
            # Should have been decomposed by Super LLM
            # This is an error condition
            decision = "escalate"
            reasoning = f"Simplicity {score:.2f} < 0.40 - should be decomposed!"
            
        # Consider Super LLM hint if present
        if query.routing.super_llm_hint:
            hint = query.routing.super_llm_hint
            if hint.confidence >= 0.80:
                # Trust Super LLM's judgment
                decision = hint.suggested_routing
                reasoning += f" | Super LLM hint: {hint.suggested_routing} (conf={hint.confidence:.2f})"
        
        return {
            "decision": decision,
            "reasoning": reasoning,
            "simplicity_score": score
        }
```

### Cluster Routing Decision

```python
class ClusterRouter:
    def route_query(self, query: QueryBlock) -> str:
        """
        Cluster decides which node(s) to use.
        
        Returns: node_id or "ensemble"
        """
        score = query.simplicity_score
        
        if score >= 0.80:
            # Very simple - single node sufficient
            return self.select_single_node(query)
        
        elif score >= 0.60:
            # Simple - might benefit from 2-node voting
            return self.select_node_pair(query)
        
        else:
            # Medium complexity - use full ensemble
            return self.select_ensemble(query)
```

---

## Query Block Lifecycle

### 1. User Query Creation

```python
def create_user_query(user_input: str, user_context: dict) -> QueryBlock:
    """
    Create initial Query Block from user input.
    """
    query_block = QueryBlock(
        query_id=generate_id("query"),
        parent_id=None,
        batch_id=None,
        depth=0,
        
        query=user_input,
        
        # Simplicity assessed by Super LLM
        simplicity_score=None,  # Will be calculated
        simplicity_reasoning=None,
        simplicity_factors=None,
        
        classification={
            "task_type": None,  # Will be determined
            "domain": None,
            "complexity": None,
            "requires_internet": False,
            "requires_calculation": False,
            "requires_code_execution": False,
            "requires_multimodal": False
        },
        
        routing={
            "current_route": "super_llm",
            "super_llm_hint": None,
            "routing_history": []
        },
        
        constraints={
            "max_latency_seconds": 60.0,
            "min_confidence": 0.70,
            "priority": "medium",
            "verification_required": True,
            "min_verifier_consensus": 0.67
        },
        
        context={
            "user_id": user_context.user_id,
            "session_id": user_context.session_id,
            "conversation_history": user_context.recent_messages,
            "timestamp": time.time()
        },
        
        metadata={
            "created_at": time.time(),
            "created_by": "user",
            "version": "1.0"
        }
    )
    
    return query_block
```

### 2. Super LLM Assessment

```python
def super_llm_assess(query: QueryBlock) -> QueryBlock:
    """
    Super LLM analyzes query and decides: answer directly or decompose?
    """
    # Calculate simplicity score
    simplicity_result = calculate_simplicity_score(
        query.query, 
        query.classification
    )
    
    query.simplicity_score = simplicity_result["simplicity_score"]
    query.simplicity_reasoning = simplicity_result["reasoning"]
    query.simplicity_factors = simplicity_result["factors"]
    
    # Classify task
    classification = super_llm_classify(query.query)
    query.classification = classification
    
    # Decision point
    if query.simplicity_score >= 0.70:
        # Simple enough - Super LLM can answer directly
        query.routing.current_route = "super_llm_direct"
        return query
    else:
        # Complex - needs decomposition
        query.routing.current_route = "decomposition"
        return query
```

### 3. Decomposition (If Complex)

```python
def decompose_query(query: QueryBlock) -> list[QueryBlock]:
    """
    Super LLM breaks complex query into simpler subtasks.
    """
    batch_id = generate_id("batch")
    
    # LLM decomposes query
    decomposition = super_llm_decompose(query.query)
    
    subtasks = []
    for i, subtask_data in enumerate(decomposition):
        # Create subtask Query Block
        subtask = QueryBlock(
            query_id=generate_id("query"),
            parent_id=query.query_id,
            batch_id=batch_id,
            depth=query.depth + 1,
            
            query=subtask_data["query"],
            
            # Super LLM provides simplicity score for each subtask
            simplicity_score=subtask_data["simplicity_score"],
            simplicity_reasoning=subtask_data["simplicity_reasoning"],
            simplicity_factors=subtask_data["factors"],
            
            classification=subtask_data["classification"],
            
            routing={
                "current_route": "super_cluster",
                "super_llm_hint": {
                    "suggested_routing": subtask_data["routing_hint"],
                    "confidence": subtask_data["routing_confidence"],
                    "reasoning": subtask_data["routing_reasoning"]
                },
                "routing_history": [
                    {
                        "timestamp": time.time(),
                        "from_block": "super-llm-main",
                        "to_block": "super-cluster-main",
                        "routing_decision": "decomposed",
                        "reasoning": "Complex query requiring decomposition"
                    }
                ]
            },
            
            constraints=query.constraints.copy(),  # Inherit from parent
            context=query.context.copy(),
            
            metadata={
                "created_at": time.time(),
                "created_by": "super-llm-main",
                "version": "1.0"
            }
        )
        
        subtasks.append(subtask)
    
    return subtasks
```

### 4. Super Cluster Routing

```python
def super_cluster_route(query: QueryBlock) -> str:
    """
    Super Cluster decides: node or cluster?
    """
    routing_decision = super_cluster_router.route_query(query)
    
    # Update routing history
    query.routing.routing_history.append({
        "timestamp": time.time(),
        "from_block": "super-cluster-main",
        "to_block": routing_decision["decision"],
        "routing_decision": routing_decision["decision"],
        "reasoning": routing_decision["reasoning"]
    })
    
    if routing_decision["decision"] == "node":
        # Find best available node
        node = super_cluster.find_optimal_node(query)
        send_to_node(node, query)
        
    elif routing_decision["decision"] == "cluster":
        # Find best cluster
        cluster = super_cluster.find_optimal_cluster(query)
        send_to_cluster(cluster, query)
        
    else:
        # Error - escalate back to Super LLM
        escalate_to_super_llm(query, "requires_decomposition")
```

### 5. Node Execution

```python
def node_execute(query: QueryBlock) -> AnswerBlock:
    """
    Node executes simple query and returns answer.
    """
    # Node's LLM generates answer
    answer = node_llm_inference(query.query)
    
    return AnswerBlock(
        answer_id=generate_id("answer"),
        query_id=query.query_id,
        answer=answer,
        confidence=calculate_confidence(answer),
        metadata={
            "node_id": node.node_id,
            "execution_time": time.time()
        }
    )
```

---

## Query Block Validation

### Validation Rules

```python
def validate_query_block(query: QueryBlock) -> bool:
    """
    Ensure Query Block is properly formed.
    """
    validations = {
        # Required fields
        "has_query_id": query.query_id is not None,
        "has_query": query.query is not None and len(query.query) > 0,
        "has_depth": query.depth is not None and query.depth >= 0,
        
        # Simplicity score if assessed
        "valid_simplicity": (
            query.simplicity_score is None or
            (0.0 <= query.simplicity_score <= 1.0)
        ),
        
        # Routing consistency
        "routing_valid": query.routing.current_route in [
            "super_llm", "super_cluster", "cluster", "node",
            "super_llm_direct", "decomposition"
        ],
        
        # Constraints
        "positive_latency": query.constraints.max_latency_seconds > 0,
        "valid_confidence": 0.0 <= query.constraints.min_confidence <= 1.0,
        
        # Context
        "has_user_id": query.context.user_id is not None,
        "has_timestamp": query.context.timestamp is not None
    }
    
    # All must pass
    return all(validations.values())
```

---

## Integration Examples

### Example 1: Simple Query (Direct to Node)

```python
# User asks simple question
user_input = "What is the capital of France?"

# Create Query Block
query = create_user_query(user_input, user_context)
# query.query = "What is the capital of France?"
# query.simplicity_score = None (not yet calculated)

# Super LLM assesses
query = super_llm_assess(query)
# query.simplicity_score = 0.78 (Simple)
# query.classification.task_type = "factual"
# query.routing.current_route = "super_llm_direct"

# Super LLM answers directly (no decomposition needed)
answer = super_llm_answer_directly(query)
# answer = "The capital of France is Paris."
```

### Example 2: Medium Query (Route to Cluster)

```python
# User asks medium-complexity question
user_input = "Find hotels in Tokyo under $150/night with good reviews"

# Create and assess
query = create_user_query(user_input, user_context)
query = super_llm_assess(query)
# query.simplicity_score = 0.62 (Medium)
# query.routing.super_llm_hint = {"suggested_routing": "cluster", ...}

# Super LLM sends to Super Cluster
subtasks = [query]  # No decomposition needed, but not simple enough for direct answer
send_to_super_cluster(subtasks)

# Super Cluster routes to cluster
super_cluster_route(query)
# Routes to "cluster-travel-specialists-007"

# Cluster uses 3-node ensemble
cluster_execute_with_ensemble(query)
```

### Example 3: Complex Query (Decompose First)

```python
# User asks complex question
user_input = "Plan a 2-week trip to Japan during cherry blossom season under $3000"

# Create and assess
query = create_user_query(user_input, user_context)
query = super_llm_assess(query)
# query.simplicity_score = 0.35 (Very Complex - Must Decompose)
# query.routing.current_route = "decomposition"

# Super LLM decomposes
subtasks = decompose_query(query)
# subtasks[0].query = "Find flights to Tokyo late March under $800"
# subtasks[0].simplicity_score = 0.62 (Medium)
# subtasks[0].routing.super_llm_hint = {"suggested_routing": "cluster"}
#
# subtasks[1].query = "Calculate remaining budget: $3000 - $750"
# subtasks[1].simplicity_score = 0.95 (Very Simple)
# subtasks[1].routing.super_llm_hint = {"suggested_routing": "node"}
#
# subtasks[2].query = "Find hotels Tokyo/Kyoto 14 nights under $1500"
# subtasks[2].simplicity_score = 0.58 (Medium)
# subtasks[2].routing.super_llm_hint = {"suggested_routing": "cluster"}
#
# subtasks[3].query = "Generate 3 alternative 14-day itineraries"
# subtasks[3].simplicity_score = 0.35 (Complex)
# subtasks[3].routing.super_llm_hint = {"suggested_routing": "cluster"}
#
# subtasks[4].query = "Check cherry blossom forecast late March"
# subtasks[4].simplicity_score = 0.78 (Simple)
# subtasks[4].routing.super_llm_hint = {"suggested_routing": "node"}

# Send batch to Super Cluster
send_to_super_cluster(subtasks)

# Super Cluster routes each according to simplicity
for subtask in subtasks:
    super_cluster_route(subtask)
    # subtask[0]: score=0.62 → cluster
    # subtask[1]: score=0.95 → node (direct)
    # subtask[2]: score=0.58 → cluster
    # subtask[3]: score=0.35 → cluster (or re-decompose if needed)
    # subtask[4]: score=0.78 → node (direct)
```

---

## Summary

The Query Block is the fundamental data structure in DELLM that:

1. **Represents questions** at every level of hierarchy
2. **Contains simplicity score** which drives ALL routing decisions
3. **Flows through system** from user → Super LLM → Super Cluster → Cluster/Node
4. **Tracks routing history** for performance analysis
5. **Maintains context** for coherent multi-turn conversations

**Key Innovation:** The simplicity score (0-1 float) is the single most important field, determining:
- Whether Super LLM answers directly or decomposes
- Whether Super Cluster routes to node or cluster
- Whether Cluster uses single node or ensemble
- How verification is performed

**Routing Rules:**
- **Simplicity ≥ 0.70** → Direct to nodes
- **0.40 ≤ Simplicity < 0.70** → Route to clusters
- **Simplicity < 0.40** → Must be decomposed by Super LLM

This architecture ensures optimal resource utilization by routing simple queries to lightweight nodes while reserving coordination overhead for queries that truly benefit from it.
