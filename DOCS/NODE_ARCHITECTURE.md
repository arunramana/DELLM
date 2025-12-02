# Node Block: Detailed Architecture Specification

## Overview

The Node Block represents an individual "species" in the DELLM ecosystem - a single LLM instance running on consumer hardware. It is the fundamental unit of execution and the primary site of evolutionary adaptation.

**CRITICAL REMINDER:** The Node Block CONTAINS an actual LLM (1B-13B parameters). This is one of only two blocks in DELLM that perform LLM inference.

---

## Node Identity & State

### Unique Identifier & Ownership
```python
NodeIdentity = {
    # Unique Node ID
    node_id: "node-550e8400-e29b-41d4-a716-446655440000",
    
    # Ownership - Critical for credit attribution
    user_id: "user-abc123",  # Person who owns this device/node
    
    # Node Type - Reflects the kind of model/specialization
    node_type: "math-specialist" | "code-specialist" | "creative-writer" | 
               "fast-generalist" | "accurate-generalist" | "logic-reasoner" |
               "factual-retriever" | "domain-expert-medical" | "domain-expert-legal",
    # Note: node_type is part of chromosome and can mutate
    
    # Network Position
    super_cluster_id: "super-cluster-main",  # Which super cluster this node belongs to
    cluster_id: "cluster-math-specialists" | "default" | null,  # Which cluster, or null if direct to super cluster
    
    # Creation Metadata
    created_at: timestamp,
    spawned_by: "new_user" | "evolution" | "cluster_request",
    parent_node_id: "node-parent-123" | null,  # Null if new user, set if spawned from evolution
    
    # Hardware Fingerprint
    device_fingerprint: hash(hardware_specs),
    device_tier: "mobile" | "low-tier" | "mid-tier" | "high-tier"
}
```

### Chromosome (Genetic Identifier)

**The chromosome is a hash/signature representing the node's complete genetic state**

**CRITICAL DESIGN PRINCIPLE: Evolution focuses on NODE TYPE and ENSEMBLE WEIGHT, NOT on tuning LLM parameters**

```python
Chromosome = {
    # Genetic Hash - Unique identifier of this exact genetic configuration
    genetic_hash: "chr-a3f5e8d9c2b1",  # Hash primarily from node_type + ensemble_weight
    generation: 7,
    
    # ========================================
    # PRIMARY EVOLUTIONARY TRAITS (80% of mutations)
    # ========================================
    
    # Node Type - The node's specialized identity
    node_type: "math_specialist",
    # Available types:
    # - "math_specialist": Mathematical reasoning, calculations, proofs
    # - "code_specialist": Programming, debugging, algorithms  
    # - "creative_specialist": Writing, brainstorming, storytelling
    # - "analytical_specialist": Data analysis, logic, research
    # - "factual_specialist": Facts, definitions, explanations
    # - "conversational_specialist": Dialogue, Q&A, general chat
    # - "scientific_specialist": Physics, chemistry, biology
    # - "business_specialist": Finance, strategy, operations
    # - "generalist": Jack-of-all-trades, balanced
    
    # Ensemble Voting Weight - How much to trust this node's answers
    ensemble_weight: 1.2,  # Range: 0.5 to 2.0
    # Weight > 1.0: This node's vote counts MORE in ensemble decisions
    # Weight = 1.0: Normal voting weight
    # Weight < 1.0: This node's vote counts LESS in ensemble decisions
    # Evolved based on fitness: high-fitness nodes get higher weights
    
    # ========================================
    # TYPE-SPECIFIC CONFIGURATION (set by node_type, rarely mutated)
    # ========================================
    
    type_config: {
        "system_prompt": "You are an expert in mathematical reasoning...",
        "few_shot_examples": [...],  # Predefined examples for this type
        "chain_of_thought": true,
        "formatting_style": "structured",
        "domain_knowledge": ["algebra", "calculus", "statistics"],
        "optimal_task_types": ["calculation", "proof", "word_problem"]
    },
    
    # ========================================
    # STATIC CONFIGURATION (locked, never mutated)
    # ========================================
    
    # Base Model (determined by hardware tier, never changes)
    base_model: {
        name: "tinyllama-1.1b-chat",  # Ultra-lightweight default: 0.6 GB
        quantization: "Q4_K_M",
        architecture: "transformer-decoder",
        size_gb: 0.6,
        ram_required_gb: 1.5
    },
    # Hardware-tier alternatives (assigned at node creation):
    # - Mobile/Tablet: tinyllama-1.1b (0.6 GB) ← DEFAULT for max participation
    # - Laptop: phi-2-2.7b (1.6 GB) ← Optional upgrade
    # - Gaming PC: phi-3-mini-3.8b (2.3 GB) ← Optional for high-end
    # NEVER use 7B+ models - reserved for Super LLM!
    
    # Fine-tuning (tied to node_type, changes when type changes)
    fine_tuning: {
        adapter: "lora-math-v2",  # Specific to node_type
        domain: "mathematical_reasoning",
        last_updated: timestamp
    },
    
    # Sampling Parameters (LOCKED to safe defaults, NEVER mutated)
    sampling_params: {
        temperature: 0.7,      # Safe default for all types
        top_p: 0.9,           # Safe default
        top_k: 50,            # Safe default
        max_tokens: 2048,     # Safe default
        repetition_penalty: 1.1,
        presence_penalty: 0.0,
        frequency_penalty: 0.0
    },
    
    # ========================================
    # LINEAGE (Evolutionary History)
    # ========================================
    
    lineage: {
        generation: 7,
        parent_chromosomes: ["chr-xyz123", "chr-abc456"] | [],
        
        mutation_history: [
            {
                generation: 5,
                mutation_type: "type_shift",  # MOST COMMON MUTATION
                old_hash: "chr-old123",
                new_hash: "chr-a3f5e8d9c2b1",
                changes: {
                    "node_type": "generalist" -> "math_specialist",
                    "ensemble_weight": 1.0 -> 1.2,
                    "fine_tuning_adapter": "lora-general-v1" -> "lora-math-v2"
                }
            },
            {
                generation: 6,
                mutation_type: "weight_adjustment",  # COMMON MUTATION
                changes: {
                    "ensemble_weight": 1.2 -> 1.5
                }
            },
            {
                generation: 7,
                mutation_type: "type_refinement",  # RARE MUTATION
                changes: {
                    "type_config.few_shot_examples": [...] -> [...]  # Better examples
                }
            }
        ]
    }
}

def compute_chromosome_hash(chromosome: Chromosome) -> str:
    """
    Compute genetic hash PRIMARILY from node_type and ensemble_weight
    LLM parameters are NOT included (they're static)
    """
    genetic_data = {
        "node_type": chromosome.node_type,
        "ensemble_weight": round(chromosome.ensemble_weight, 2),
        "generation": chromosome.lineage.generation,
        "fine_tuning": chromosome.fine_tuning.adapter
    }
    return f"chr-{hash(json.dumps(genetic_data, sort_keys=True))[:12]}"
```

### Node Type Definitions

**Each node type has a predefined configuration optimized for its domain:**

```python
NODE_TYPE_CONFIGS = {
    "math_specialist": {
        "system_prompt": """You are an expert in mathematical reasoning. 
        Approach problems step-by-step, show your work, and verify calculations.
        Focus on precision and correctness over speed.""",
        
        "few_shot_examples": [
            {
                "query": "What is 15% of 240?",
                "answer": "Let me solve this step by step:\n1. Convert 15% to decimal: 15/100 = 0.15\n2. Multiply: 0.15 × 240 = 36\nAnswer: 36"
            },
            {
                "query": "Solve for x: 2x + 5 = 13",
                "answer": "Step by step:\n1. Subtract 5 from both sides: 2x = 8\n2. Divide both sides by 2: x = 4\nAnswer: x = 4"
            }
        ],
        
        "chain_of_thought": true,
        "formatting_style": "structured",
        "domain_knowledge": ["algebra", "calculus", "statistics", "geometry"],
        "optimal_task_types": ["calculation", "proof", "word_problem", "equation_solving"],
        "default_ensemble_weight": 1.0,
        "fine_tuning_adapter": "lora-math-v2"
    },
    
    "code_specialist": {
        "system_prompt": """You are an expert programmer. 
        Write clean, efficient, well-documented code.
        Debug systematically and explain your reasoning.""",
        
        "few_shot_examples": [
            {
                "query": "Write a function to find prime numbers",
                "answer": "```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```"
            }
        ],
        
        "chain_of_thought": true,
        "formatting_style": "code_blocks",
        "domain_knowledge": ["python", "javascript", "algorithms", "debugging"],
        "optimal_task_types": ["code_generation", "debugging", "code_review", "algorithm_design"],
        "default_ensemble_weight": 1.0,
        "fine_tuning_adapter": "lora-code-v2"
    },
    
    "creative_specialist": {
        "system_prompt": """You are a creative writer and brainstormer.
        Generate original, engaging content with vivid details.
        Think outside the box and explore multiple perspectives.""",
        
        "few_shot_examples": [
            {
                "query": "Write a short story opening about a detective",
                "answer": "The rain hammered against the window as Detective Sarah Chen studied the photograph. Three victims, three cities, three months. But only one connection: a single chess piece left at each scene..."
            }
        ],
        
        "chain_of_thought": false,
        "formatting_style": "narrative",
        "domain_knowledge": ["storytelling", "poetry", "marketing", "ideation"],
        "optimal_task_types": ["creative_writing", "brainstorming", "marketing_copy", "storytelling"],
        "default_ensemble_weight": 1.0,
        "fine_tuning_adapter": "lora-creative-v1"
    },
    
    "analytical_specialist": {
        "system_prompt": """You are a data analyst and logical reasoner.
        Break down complex information, identify patterns, draw evidence-based conclusions.
        Be precise, objective, and thorough.""",
        
        "few_shot_examples": [...],
        "chain_of_thought": true,
        "formatting_style": "structured",
        "domain_knowledge": ["data_analysis", "logic", "research", "critical_thinking"],
        "optimal_task_types": ["analysis", "comparison", "research", "reasoning"],
        "default_ensemble_weight": 1.0,
        "fine_tuning_adapter": "lora-analytical-v1"
    },
    
    "factual_specialist": {
        "system_prompt": """You are a knowledge expert focused on accurate information.
        Provide clear, concise, factual answers. Cite knowledge when possible.
        Admit uncertainty rather than guess.""",
        
        "few_shot_examples": [...],
        "chain_of_thought": false,
        "formatting_style": "concise",
        "domain_knowledge": ["history", "science", "geography", "definitions"],
        "optimal_task_types": ["factual_query", "definition", "explanation", "trivia"],
        "default_ensemble_weight": 1.0,
        "fine_tuning_adapter": "lora-factual-v1"
    },
    
    "conversational_specialist": {
        "system_prompt": """You are a helpful conversational assistant.
        Be friendly, natural, and engaging. Adapt your tone to the user.
        Focus on clarity and helpfulness.""",
        
        "few_shot_examples": [...],
        "chain_of_thought": false,
        "formatting_style": "conversational",
        "domain_knowledge": ["general_knowledge", "common_sense", "advice"],
        "optimal_task_types": ["general_chat", "advice", "how_to", "explanation"],
        "default_ensemble_weight": 1.0,
        "fine_tuning_adapter": "lora-conversational-v1"
    },
    
    "generalist": {
        "system_prompt": """You are a helpful AI assistant capable of handling diverse tasks.
        Adapt your approach based on the question type.
        Strive for accuracy and clarity.""",
        
        "few_shot_examples": [...],
        "chain_of_thought": true,
        "formatting_style": "adaptive",
        "domain_knowledge": ["general"],
        "optimal_task_types": ["general"],
        "default_ensemble_weight": 1.0,
        "fine_tuning_adapter": "lora-general-v1"
    }
}
```
        
        # Track node_type changes through evolution
        type_evolution: [
            {
                generation: 1,
                old_type: null,  # Genesis
                new_type: "fast-generalist"
            },
            {
                generation: 5,
                old_type: "fast-generalist",
                new_type: "math-specialist",  # Specialized via mutation
                reason: "high_fitness_on_math_tasks"
            }
        ],
        
        mutation_history: [
            {
                generation: 5,
                mutation_type: "type_specialization",  # New mutation type
                old_hash: "chr-gen-old123",
                new_hash: "chr-math-a3f5e8d9c2b1",
                changes: {
                    "node_type": "fast-generalist" -> "math-specialist",
                    "temperature": 0.7 -> 0.3,
                    "system_prompt": "You are a helpful assistant" -> "You are a mathematical reasoning expert",
                    "fine_tuning_adapter": "general-v1" -> "lora-math-v2"
                }
            },
            {
                generation: 6,
                mutation_type: "parameter_tweak",
                old_hash: "chr-math-a3f5e8d9c2b1",
                new_hash: "chr-math-b7k2m9f4e8x3",
                changes: {"temperature": 0.3 -> 0.35}
            }
        ]
    }
}

def compute_chromosome_hash(chromosome: Chromosome) -> str:
    """
    Compute genetic hash from ALL parameters INCLUDING node_type
    Any change to node_type = new chromosome hash
    """
    genetic_data = {
        "node_type": chromosome.node_type.primary_type,  # CRITICAL: Type is part of hash
        "type_signature": chromosome.node_type.type_signature,
        "base_model": chromosome.base_model.name,
        "fine_tuning": chromosome.fine_tuning.adapter,
        "system_prompt": chromosome.prompt_template.system_prompt,
        "temperature": chromosome.sampling_params.temperature,
        "top_p": chromosome.sampling_params.top_p,
        "top_k": chromosome.sampling_params.top_k,
        "specialization": chromosome.specialization.primary_domain
    }
    
    # Hash includes node_type, so type changes = different hash
    hash_value = hash(json.dumps(genetic_data, sort_keys=True))
    
    # Prefix hash with type identifier for easy recognition
    type_prefix = {
        "math-specialist": "math",
        "code-specialist": "code",
        "creative-writer": "crea",
        "fast-generalist": "fgen",
        "accurate-generalist": "agen",
        "logic-reasoner": "logi",
        "factual-retriever": "fact"
    }.get(chromosome.node_type.primary_type, "unkn")
    
    return f"chr-{type_prefix}-{abs(hash_value) % 10**12:012d}"
```

### Node Type Archetypes

Each node type has a distinct genetic profile:

```python
NODE_TYPE_ARCHETYPES = {
    "math-specialist": {
        "temperature": 0.2,  # Very precise
        "system_prompt": "You are a mathematical reasoning expert. Solve problems step-by-step with precision.",
        "fine_tuning_adapter": "lora-math-reasoning",
        "primary_domain": "mathematical_reasoning",
        "chain_of_thought": True,
        "max_tokens": 2048
    },
    
    "code-specialist": {
        "temperature": 0.3,
        "system_prompt": "You are an expert programmer. Write clean, efficient, well-documented code.",
        "fine_tuning_adapter": "lora-code-generation",
        "primary_domain": "programming",
        "chain_of_thought": True,
        "max_tokens": 4096
    },
    
    "creative-writer": {
        "temperature": 0.9,  # High creativity
        "system_prompt": "You are a creative writer. Generate engaging, original content.",
        "fine_tuning_adapter": "lora-creative-writing",
        "primary_domain": "creative_content",
        "chain_of_thought": False,
        "max_tokens": 2048
    },
    
    "fast-generalist": {
        "temperature": 0.5,
        "system_prompt": "You are a helpful assistant. Provide quick, accurate answers.",
        "fine_tuning_adapter": "lora-general-fast",
        "primary_domain": "general",
        "chain_of_thought": False,
        "max_tokens": 1024  # Shorter for speed
    },
    
    "accurate-generalist": {
        "temperature": 0.3,
        "system_prompt": "You are a helpful assistant. Provide thorough, well-reasoned answers.",
        "fine_tuning_adapter": "lora-general-accurate",
        "primary_domain": "general",
        "chain_of_thought": True,
        "max_tokens": 2048
    },
    
    "logic-reasoner": {
        "temperature": 0.2,
        "system_prompt": "You are a logical reasoning expert. Deduce conclusions from premises rigorously.",
        "fine_tuning_adapter": "lora-logical-reasoning",
        "primary_domain": "logical_reasoning",
        "chain_of_thought": True,
        "max_tokens": 2048
    },
    
    "factual-retriever": {
        "temperature": 0.1,  # Very low for factual accuracy
        "system_prompt": "You are a fact-checking expert. Provide precise, verifiable information.",
        "fine_tuning_adapter": "lora-factual-qa",
        "primary_domain": "factual_knowledge",
        "chain_of_thought": False,
        "max_tokens": 1024
    },
    
    "domain-expert-medical": {
        "temperature": 0.2,
        "system_prompt": "You are a medical knowledge expert. Provide evidence-based medical information.",
        "fine_tuning_adapter": "lora-medical-domain",
        "primary_domain": "medical",
        "chain_of_thought": True,
        "max_tokens": 3072
    }
}
```

### Node Type Genetic Profiles Comparison

**How node_type influences ALL genetic parameters:**

| Parameter | math-specialist | code-specialist | creative-writer | fast-generalist | accurate-generalist | logic-reasoner |
|-----------|----------------|----------------|----------------|----------------|-------------------|---------------|
| **Temperature** | 0.2 (precise) | 0.3 | 0.9 (creative) | 0.5 | 0.3 | 0.2 (precise) |
| **LoRA Adapter** | lora-math-v2 | lora-code-gen | lora-creative | lora-gen-fast | lora-gen-accurate | lora-logic |
| **Primary Domain** | mathematical | programming | creative | general | general | logical |
| **Chain-of-Thought** | ✅ Yes | ✅ Yes | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| **Max Tokens** | 2048 | 4096 | 2048 | 1024 | 2048 | 2048 |
| **Top-P** | 0.9 | 0.9 | 0.95 | 0.9 | 0.9 | 0.85 |
| **Top-K** | 40 | 50 | 100 | 50 | 40 | 30 |
| **System Prompt** | "Math expert..." | "Expert programmer..." | "Creative writer..." | "Quick assistant..." | "Thorough assistant..." | "Logic expert..." |

**Chromosome hash includes node_type:**
```
math-specialist with temp=0.2 → chr-math-a3f5e8d9c2b1
code-specialist with temp=0.3 → chr-code-b7k2m9f4e8x3
creative-writer with temp=0.9 → chr-crea-f9x1p4k7m2b5

Same parameters but different type = different hash!
```

### Type Evolution Examples

**Example 1: Generalist discovers talent in math**
```python
Generation 1: fast-generalist (chr-fgen-123abc)
    ↓ (performs exceptionally well on math tasks)
Generation 5: math-specialist (chr-math-789xyz)

Changes:
- node_type: "fast-generalist" → "math-specialist"
- temperature: 0.5 → 0.2
- system_prompt: "helpful assistant" → "mathematical reasoning expert"
- fine_tuning_adapter: "lora-general-fast" → "lora-math-v2"
- max_tokens: 1024 → 2048
- chain_of_thought: False → True
```

**Example 2: Cluster rebalancing (too many math specialists)**
```python
Generation 8: math-specialist (chr-math-789xyz)
    ↓ (cluster needs code specialists)
Generation 9: code-specialist (chr-code-456def)

Changes:
- node_type: "math-specialist" → "code-specialist"
- temperature: 0.2 → 0.3
- system_prompt: "mathematical expert" → "expert programmer"
- fine_tuning_adapter: "lora-math-v2" → "lora-code-generation"
- primary_domain: "mathematical_reasoning" → "programming"
- max_tokens: 2048 → 4096
```

**Example 3: Crossover inherits type from fitter parent**
```python
Parent A: math-specialist (fitness: 0.85, chr-math-111aaa)
Parent B: logic-reasoner (fitness: 0.72, chr-logi-222bbb)
    ↓ (crossover)
Child: math-specialist (chr-math-333ccc)

Inherited:
- node_type: "math-specialist" (from Parent A - higher fitness)
- temperature: 0.2 (from Parent A)
- top_p: 0.85 (from Parent B - genetic diversity)
- few_shot_examples: mixed from both parents
```

### Runtime State
```python
RuntimeState = {
    # Execution Status
    status: "idle" | "executing" | "mutating" | "killed" | "offline",
    current_task: TaskID | null,
    
    # Fitness Score - CRITICAL METRIC
    fitness: {
        current_score: 0.82,  # Current fitness (0.0 to 1.0)
        score_history: [
            {"timestamp": t1, "score": 0.78},
            {"timestamp": t2, "score": 0.80},
            {"timestamp": t3, "score": 0.82}
        ],
        tasks_evaluated: 147,  # Number of tasks used to compute fitness
        last_updated: timestamp
    },
    
    # Performance Metrics (used to compute fitness)
    performance: {
        total_tasks_completed: 1247,
        total_tasks_failed: 23,
        recent_correctness: 0.87,    # Recent average correctness (weighted)
        recent_latency: 2.3,          # Recent average latency in seconds (weighted)
        uptime_hours: 482.5
    },
    
    # Resource Monitoring
    resources: {
        cpu_usage: 45.2,
        memory_usage: 12.5,
        gpu_usage: 78.3 | null,
        thermal_state: "normal" | "warm" | "hot",
        battery_level: 85 | null  # For mobile devices
    },
    
    # Credit Economics (belongs to user_id)
    credits_contributed: {
        balance: 15420,  # Credits earned by this node for the user
        earned_lifetime: 20000,
        earning_rate: 45.2  # Credits per hour (current)
    },
    
    # Network Connectivity
    network: {
        connected_to_super_cluster: true,
        connected_to_cluster: true | false,  # False if cluster_id is null
        latency_to_super_cluster_ms: 45
    },
    
    # Current Chromosome
    active_chromosome: "chr-a3f5e8d9c2b1"  # Reference to current chromosome hash
}
```

---

## Input Interfaces

### 1. Task Assignment (from Super Cluster or Cluster)

**Primary Input:** Receive simple query to answer

```python
TaskAssignment = {
    # Task Identification
    task_id: "task-abc123",
    parent_query_id: "query-xyz789",
    assigned_by_cluster: "cluster-math-specialists" | null,  # Null if from super cluster directly
    assigned_by_super_cluster: "super-cluster-main",
    priority: "low" | "medium" | "high",
    
    # The Simple Query (this is what the node answers)
    query: "Calculate the compound interest on $10,000 at 5% annually for 10 years",
    
    # Task Classification (for routing/fitness evaluation)
    classification: {
        task_type: "calculation" | "reasoning" | "creative" | "factual",
        domain: "mathematics" | "programming" | "general",
        complexity: "simple" | "medium" | "complex"
    },
    
    # Execution Requirements
    constraints: {
        max_latency_seconds: 5.0,
        max_tokens: 500,
        min_confidence: 0.8
    },
    
    # Redundancy Setup (node runs 2 LLMs in parallel internally)
    internal_redundancy: {
        num_parallel_runs: 2,  # Always 2 for voting
        voting_strategy: "majority" | "highest_confidence"
    },
    
    # Metadata
    timestamp: 1640995200,
    timeout_seconds: 10.0
}
```

**Input Channel:** gRPC stream from Super Cluster or Cluster coordinator

**Acceptance Logic:**
```python
def should_accept_task(task: TaskAssignment) -> bool:
    # Check if node is alive (not killed)
    if runtime_state.status == "killed":
        return False
    
    # Check resource availability
    if runtime_state.resources.cpu_usage > 90:
        return False
    
    # Check thermal state (important for mobile devices)
    if runtime_state.resources.thermal_state == "hot":
        return False
    
    # Check battery (for mobile devices)
    if runtime_state.resources.battery_level and runtime_state.resources.battery_level < 20:
        return False
    
    return True
```

### 2. Evolutionary Updates (from Cluster)

**Input:** Receive genetic modifications during evolution cycles

```python
EvolutionaryUpdate = {
    update_id: "evo-update-789",
    update_type: "mutation" | "crossover" | "type_specialization" | "type_shift" | "fine_tune",
    generation: 8,  # New generation number
    
    # Parameter Mutation (minor tweaks)
    mutation: {
        parameter_changes: {
            "temperature": 0.3 -> 0.35,
            "top_p": 0.9 -> 0.92
        },
        prompt_changes: {
            "system_prompt": "New optimized prompt...",
            "add_examples": [...]
        }
    } | null,
    
    # Type Specialization (generalist → specialist)
    type_specialization: {
        old_type: "fast-generalist",
        new_type: "math-specialist",
        reason: "high_fitness_on_math_tasks",  # Cluster observed node excels at math
        
        # Complete genetic overhaul to match new type archetype
        apply_archetype: NODE_TYPE_ARCHETYPES["math-specialist"],
        
        # Changes triggered by type specialization
        cascading_changes: {
            "temperature": 0.5 -> 0.2,
            "system_prompt": "You are a helpful assistant" -> "You are a mathematical reasoning expert",
            "fine_tuning_adapter": "lora-general-fast" -> "lora-math-v2",
            "primary_domain": "general" -> "mathematical_reasoning",
            "max_tokens": 1024 -> 2048
        }
    } | null,
    
    # Type Shift (specialist → different specialist or specialist → generalist)
    type_shift: {
        old_type: "math-specialist",
        new_type: "code-specialist",
        reason: "cluster_needs_rebalancing",  # Too many math specialists, need code specialists
        
        apply_archetype: NODE_TYPE_ARCHETYPES["code-specialist"],
        
        cascading_changes: {
            "temperature": 0.2 -> 0.3,
            "system_prompt": "You are a mathematical reasoning expert" -> "You are an expert programmer",
            "fine_tuning_adapter": "lora-math-v2" -> "lora-code-generation",
            "primary_domain": "mathematical_reasoning" -> "programming"
        }
    } | null,
    
    # Crossover Update (genes from two parents)
    crossover: {
        parent_a: "node-123",
        parent_a_type: "math-specialist",
        parent_b: "node-456",
        parent_b_type: "logic-reasoner",
        
        # Child inherits type from higher-fitness parent
        child_type: "math-specialist",  # Parent A had higher fitness
        
        inherited_params: {
            from_a: {
                "node_type": "math-specialist",
                "temperature": 0.2,
                "system_prompt": "..."
            },
            from_b: {
                "top_p": 0.95,
                "few_shot_examples": [...],
                "chain_of_thought": true
            }
        }
    } | null,
    
    # Fine-tuning Update
    fine_tune: {
        adapter_update: "lora-math-v3",  # New LoRA adapter
        training_data: [...],  # Recent successful task history
        training_config: {...},
        # Note: Fine-tuning does NOT change node_type, just improves it
    } | null,
    
    # Metadata
    reason: "performance_improvement" | "specialization_deepening" | 
            "diversity_maintenance" | "cluster_rebalancing",
    expected_improvement: 0.15,  # Expected fitness increase
    rollback_available: true
}
```

**Input Channel:** gRPC RPC call from Cluster coordinator

**Application Logic:**
```python
def apply_evolutionary_update(update: EvolutionaryUpdate):
    # Backup current chromosome (for potential rollback)
    chromosome_backup = deepcopy(chromosome)
    old_chromosome_hash = chromosome.genetic_hash
    
    # Apply simple mutation (parameter tweaks)
    if update.mutation:
        chromosome.sampling_params.update(update.mutation.parameter_changes)
        chromosome.prompt_template.update(update.mutation.prompt_changes)
    
    # Apply type specialization (MAJOR genetic change)
    if update.type_specialization:
        old_type = chromosome.node_type.primary_type
        new_type = update.type_specialization.new_type
        
        # Update node type
        chromosome.node_type.primary_type = new_type
        node_identity.node_type = new_type  # Sync with identity
        
        # Apply entire archetype (major genetic overhaul)
        archetype = update.type_specialization.apply_archetype
        chromosome.sampling_params.temperature = archetype["temperature"]
        chromosome.prompt_template.system_prompt = archetype["system_prompt"]
        chromosome.fine_tuning.adapter = archetype["fine_tuning_adapter"]
        chromosome.specialization.primary_domain = archetype["primary_domain"]
        chromosome.prompt_template.chain_of_thought = archetype["chain_of_thought"]
        chromosome.sampling_params.max_tokens = archetype["max_tokens"]
        
        # Track type evolution
        chromosome.lineage.type_evolution.append({
            "generation": update.generation,
            "old_type": old_type,
            "new_type": new_type,
            "reason": update.type_specialization.reason
        })
        
        # Reload model with new fine-tuning adapter
        runtime_state.status = "mutating"
        load_lora_adapter(archetype["fine_tuning_adapter"])
    
    # Apply type shift (specialist → different specialist)
    if update.type_shift:
        old_type = chromosome.node_type.primary_type
        new_type = update.type_shift.new_type
        
        # Similar to type_specialization but often for cluster balancing
        chromosome.node_type.primary_type = new_type
        node_identity.node_type = new_type
        
        archetype = update.type_shift.apply_archetype
        # Apply archetype changes...
        
        chromosome.lineage.type_evolution.append({
            "generation": update.generation,
            "old_type": old_type,
            "new_type": new_type,
            "reason": update.type_shift.reason
        })
    
    # Apply crossover
    if update.crossover:
        # Child inherits type from one parent (usually higher-fitness parent)
        chromosome.node_type.primary_type = update.crossover.child_type
        node_identity.node_type = update.crossover.child_type
        
        chromosome = merge_genomes(
            chromosome,
            update.crossover.inherited_params
        )
    
    # Apply fine-tuning (requires model reload)
    if update.fine_tune:
        runtime_state.status = "mutating"
        load_lora_adapter(update.fine_tune.adapter_update)
        chromosome.fine_tuning.adapter = update.fine_tune.adapter_update
    
    # Compute new chromosome hash (includes node_type)
    new_chromosome_hash = compute_chromosome_hash(chromosome)
    chromosome.genetic_hash = new_chromosome_hash
    runtime_state.active_chromosome = new_chromosome_hash
    
    # Update lineage tracking
    chromosome.lineage.generation = update.generation
    chromosome.lineage.mutation_history.append({
        "generation": update.generation,
        "update_type": update.update_type,
        "old_hash": old_chromosome_hash,
        "new_hash": new_chromosome_hash,
        "changes": update.mutation | update.type_specialization | update.type_shift | update.crossover
    })
    
    # Reset fitness for new chromosome
    runtime_state.fitness.current_score = 0.5
    runtime_state.fitness.tasks_evaluated = 0
    
    # Warm up model with new parameters
    runtime_state.status = "idle"
    warmup_model()
```

### 3. Fitness Update (from Super Cluster or Cluster)

**Input:** Receive fitness score update after verification

```python
FitnessUpdate = {
    update_id: "fitness-update-789",
    task_id: "task-abc123",
    node_id: "node-550e8400",
    
    # Verification Outcome
    verification: {
        correctness: 0.95,  # 0.0 = wrong, 1.0 = perfect
        verified_by: "super-cluster-main",
        verification_method: "programmatic" | "redundancy" | "super_llm",
        latency_actual: 2.3  # seconds (measured by cluster/super cluster)
    },
    
    # New Fitness Score Calculation
    fitness_calculation: {
        # Formula: weighted_correctness / (1 + latency_normalized)
        correctness_weight: 0.7,  # How much correctness matters
        latency_weight: 0.3,      # How much speed matters
        
        # Normalized metrics
        correctness_normalized: 0.95,
        latency_normalized: 0.23,  # Lower is better, normalized 0-1
        
        # Raw fitness for this task
        task_fitness: 0.77,  # = 0.95 / (1 + 0.23)
        
        # Updated node fitness (weighted average with history)
        old_fitness_score: 0.82,
        new_fitness_score: 0.81,  # = 0.9 * old + 0.1 * task_fitness
        weight_old: 0.9,  # How much to keep old score
        weight_new: 0.1   # How much to incorporate new score
    },
    
    # Fitness Ranking
    ranking: {
        cluster_average_fitness: 0.75,
        percentile_in_cluster: 85,  # This node is top 15%
        rank_in_cluster: 12,  # Out of 80 nodes
        signal: "positive" | "neutral" | "negative"  # Positive if above average
    },
    
    # Credit Earned
    credits_earned: {
        amount: 45,
        calculation: {
            base_rate: 30,
            quality_multiplier: 1.27,  # Based on correctness = 0.95
            latency_bonus: 1.05,  # Fast response
            difficulty_multiplier: 1.1
        }
    },
    
    # Update Instructions
    update_state: {
        new_fitness_score: 0.81,
        tasks_evaluated_count: 148,  # Increment by 1
        update_chromosome_fitness: true  # This chromosome's fitness improves
    }
}
```

**Input Channel:** gRPC RPC from Super Cluster or Cluster

**State Update Logic:**
```python
def apply_fitness_update(update: FitnessUpdate):
    """
    Update node's fitness score based on verification
    """
    # Update fitness score (weighted average)
    old_score = runtime_state.fitness.current_score
    new_score = update.fitness_calculation.new_fitness_score
    
    runtime_state.fitness.current_score = new_score
    runtime_state.fitness.tasks_evaluated += 1
    runtime_state.fitness.last_updated = time.now()
    
    # Add to score history
    runtime_state.fitness.score_history.append({
        "timestamp": time.now(),
        "score": new_score,
        "task_id": update.task_id
    })
    
    # Keep only last 100 scores in history
    if len(runtime_state.fitness.score_history) > 100:
        runtime_state.fitness.score_history = runtime_state.fitness.score_history[-100:]
    
    # Update performance metrics
    runtime_state.performance.total_tasks_completed += 1
    
    # Update recent correctness (exponential moving average)
    alpha = 0.1
    runtime_state.performance.recent_correctness = (
        (1 - alpha) * runtime_state.performance.recent_correctness +
        alpha * update.verification.correctness
    )
    
    # Update recent latency (exponential moving average)
    runtime_state.performance.recent_latency = (
        (1 - alpha) * runtime_state.performance.recent_latency +
        alpha * update.verification.latency_actual
    )
    
    # Update credits
    runtime_state.credits_contributed.balance += update.credits_earned.amount
    runtime_state.credits_contributed.earned_lifetime += update.credits_earned.amount
```

### 4. Credit Transactions (from Credit System)

**Input:** Credit earnings and spending notifications

```python
CreditTransaction = {
    transaction_id: "txn-123",
    type: "earn" | "spend" | "bonus" | "penalty",
    
    # Earnings (for task completion)
    earn: {
        amount: 45,
        task_id: "task-abc123",
        calculation: {
            base_rate: 30,           # credits per task
            quality_multiplier: 1.2, # based on correctness
            uptime_bonus: 1.1,       # reliable node
            difficulty_factor: 1.05  # harder task = more credits
        }
    } | null,
    
    # Spending (for query submission - future feature)
    spend: {
        amount: 120,
        query_id: "query-xyz789",
        service: "complex_query_execution"
    } | null,
    
    # Bonus (special events)
    bonus: {
        amount: 100,
        reason: "new_user_welcome" | "referral" | "milestone_reached"
    } | null,
    
    # Penalty (misbehavior)
    penalty: {
        amount: 50,
        reason: "incorrect_answer_pattern" | "excessive_downtime"
    } | null,
    
    # New Balance
    new_balance: 15420,
    timestamp: 1640995200
}
```

**Input Channel:** REST API or gRPC from Credit Management Service

---

## Output Interfaces

### 1. Task Results (to Super Cluster or Cluster)

**Primary Output:** Return answer after running 2 LLMs in parallel and voting

```python
TaskResult = {
    # Identification
    task_id: "task-abc123",
    node_id: "node-550e8400",
    user_id: "user-abc123",  # For credit attribution
    cluster_id: "cluster-math-specialists" | null,
    super_cluster_id: "super-cluster-main",
    
    # Genetic State
    chromosome: "chr-a3f5e8d9c2b1",  # Genetic state that produced this answer
    node_type: "math_specialist",    # Type of node that answered
    ensemble_weight: 1.2,             # How much to trust this answer in voting
    
    # Parallel Execution Results
    parallel_runs: [
        {
            run_id: 1,
            answer: "The compound interest is $16,288.95",
            confidence: 0.92,
            latency_seconds: 2.1,
            tokens_generated: 87
        },
        {
            run_id: 2,
            answer: "The compound interest is $16,288.95",
            confidence: 0.89,
            latency_seconds: 2.3,
            tokens_generated: 91
        }
    ],
    
    # Voting Result (with ensemble weighting)
    voted_answer: {
        answer: "The compound interest is $16,288.95",
        vote_agreement: "unanimous" | "majority" | "split",
        confidence: 0.905,  # Raw confidence (average of agreeing runs)
        ensemble_weighted_confidence: 1.086,  # confidence × ensemble_weight
        reasoning: "Both LLM runs produced identical answer (ensemble weight: 1.2)"
    },
    
    # Execution Metadata
    execution: {
        total_latency_seconds: 2.3,  # Max of parallel runs
        total_tokens_generated: 178,  # Sum of both runs
        compute_cost: 0.084  # Normalized compute units for both runs
    },
    
    # Send to Cluster or Super Cluster for verification
    send_to: {
        cluster_id: "cluster-math-specialists" | null,
        super_cluster_id: "super-cluster-main"
    },
    
    # Timestamp
    timestamp: 1640995202.3
}
```

**Output Channel:** gRPC response to Cluster (if part of cluster) AND Super Cluster

**Key Point:** The `ensemble_weight` allows Cluster/Super Cluster to weight this node's vote higher or lower when combining answers from multiple nodes

**Voting Logic:**
```python
def vote_on_answers(run1: LLMResult, run2: LLMResult) -> VotedAnswer:
    """
    Compare two parallel LLM runs and vote on final answer
    """
    # Exact match - unanimous
    if run1.answer == run2.answer:
        return VotedAnswer(
            answer=run1.answer,
            vote_agreement="unanimous",
            confidence=(run1.confidence + run2.confidence) / 2,
            reasoning="Both LLM runs produced identical answer"
        )
    
    # Different answers - use confidence to break tie
    elif run1.confidence > run2.confidence:
        return VotedAnswer(
            answer=run1.answer,
            vote_agreement="majority",
            confidence=run1.confidence,
            reasoning=f"Run 1 had higher confidence ({run1.confidence} vs {run2.confidence})"
        )
    else:
        return VotedAnswer(
            answer=run2.answer,
            vote_agreement="majority",
            confidence=run2.confidence,
            reasoning=f"Run 2 had higher confidence ({run2.confidence} vs {run1.confidence})"
        )
```

### 2. Performance Reports (to Cluster and Super Cluster)

**Output:** Regular performance and status updates

```python
PerformanceReport = {
    # Identification
    node_id: "node-550e8400",
    report_type: "periodic" | "on_demand" | "critical",
    reporting_period: {
        start: 1640995200,
        end: 1640998800,
        duration_hours: 1.0
    },
    
    # Task Execution Statistics
    tasks: {
        completed: 47,
        failed: 2,
        rejected: 5,  # Tasks declined due to load/specialization
        success_rate: 0.96,
        
        # By Task Type
        by_type: {
            "calculation": {completed: 30, avg_correctness: 0.94},
            "word_problems": {completed: 15, avg_correctness: 0.89},
            "proofs": {completed: 2, avg_correctness: 0.75}
        }
    },
    
    # Performance Metrics
    performance: {
        average_latency: 2.3,
        average_correctness: 0.91,
        average_confidence: 0.87,
        throughput: 47  # tasks per hour
    },
    
    # Fitness Score (Self-Calculated)
    fitness: {
        current: 0.82,
        formula: "(correctness × speed) / (latency + compute_cost)",
        trend: "improving" | "stable" | "declining",
        change_since_last: +0.05
    },
    
    # Resource Utilization
    resources: {
        average_cpu: 52.3,
        average_memory: 11.2,
        average_gpu: 68.5,
        peak_cpu: 89.2,
        peak_memory: 14.1,
        uptime_percentage: 98.5
    },
    
    # Specialization Performance
    specialization_efficacy: {
        primary_domain_accuracy: 0.94,   # Math
        secondary_domain_accuracy: 0.88, # Logic
        general_accuracy: 0.82,          # Other tasks
        
        # Recommendation: specialize further if primary >> general
        specialization_delta: 0.12
    },
    
    # Credit Economics
    credits_earned: 2140,
    credits_spent: 0,
    earning_rate: 2140  # per hour
}
```

**Output Channel:** gRPC RPC call to Cluster coordinator (periodic, every 30-60 minutes)

### 3. Availability Broadcasts (to Super Cluster)

**Output:** Announce availability for task assignment

```python
AvailabilityBroadcast = {
    # Identification
    node_id: "node-550e8400",
    status: "available" | "busy" | "maintenance" | "offline",
    
    # Current State
    current_load: {
        cpu_usage: 45.2,
        queue_depth: 2,  # Tasks in local queue
        estimated_available_in: 30  # seconds until next task slot
    },
    
    # Capabilities
    capabilities: {
        specializations: ["mathematical_reasoning", "symbolic_logic"],
        max_tokens: 2048,
        supports_code_execution: false,
        supports_image_generation: false
    },
    
    # Constraints
    constraints: {
        battery_level: 85,
        thermal_state: "normal",
        willing_to_accept: ["calculation", "proofs", "word_problems"],
        unwilling_to_accept: ["creative_writing"]  # Low fitness here
    },
    
    # Network Info
    network: {
        cluster_memberships: ["cluster-math-specialists"],
        latency_to_super_cluster: 45,  # ms
        bandwidth_available: 100  # Mbps
    }
}
```

**Output Channel:** gRPC stream or heartbeat to Super Cluster (every 10-30 seconds)

### 4. Evolution Feedback (to Cluster)

**Output:** Report outcomes of evolutionary updates

```python
EvolutionFeedback = {
    # Identification
    update_id: "evo-update-789",
    node_id: "node-550e8400",
    
    # Application Result
    application: {
        status: "success" | "failed" | "rolled_back",
        applied_at: 1640995200,
        error_message: null | "..."
    },
    
    # Performance Impact (after applying update)
    impact: {
        fitness_before: 0.78,
        fitness_after: 0.82,
        delta: +0.04,
        
        # Measured over N tasks after update
        measurement_window: {
            tasks_completed: 25,
            start_time: 1640995200,
            end_time: 1640997000
        },
        
        # Breakdown
        correctness_change: +0.03,
        latency_change: -0.2,  # seconds (improvement)
        confidence_change: +0.05
    },
    
    # Recommendation
    recommendation: "keep" | "rollback" | "continue_testing",
    
    # Additional Observations
    observations: {
        specialization_improved: true,
        general_performance_decreased: false,
        unexpected_side_effects: []
    }
}
```

**Output Channel:** gRPC RPC call to Cluster coordinator (sent after sufficient post-update testing)

---

## Internal Mechanisms

### 1. Task Execution Pipeline

**Core Process: Run 2 LLMs in parallel, vote on answer, send to cluster/super cluster**

```python
class TaskExecutor:
    def execute_task(self, task: TaskAssignment) -> TaskResult:
        """
        Main task execution: parallel LLM runs + voting
        """
        start_time = time.now()
        
        # Step 1: Validate task acceptance
        if not self.should_accept_task(task):
            return TaskRejection(reason="resource_constraints")
        
        # Step 2: Update status
        self.runtime_state.status = "executing"
        self.runtime_state.current_task = task.task_id
        
        # Step 3: Prepare prompt using chromosome template
        prompt = self.format_prompt(
            query=task.query,
            template=self.chromosome.prompt_template
        )
        
        # Step 4: Run 2 LLMs in parallel ⚡⚡
        parallel_runs = await asyncio.gather(
            self.run_llm(prompt, run_id=1),
            self.run_llm(prompt, run_id=2)
        )
        
        # Step 5: Vote on the two answers
        voted_answer = self.vote_on_answers(parallel_runs[0], parallel_runs[1])
        
        # Step 6: Package result
        result = TaskResult(
            task_id=task.task_id,
            node_id=self.node_id,
            user_id=self.user_id,
            cluster_id=self.cluster_id,
            super_cluster_id=self.super_cluster_id,
            chromosome=self.runtime_state.active_chromosome,
            parallel_runs=[
                {
                    "run_id": 1,
                    "answer": parallel_runs[0].answer,
                    "confidence": parallel_runs[0].confidence,
                    "latency_seconds": parallel_runs[0].latency,
                    "tokens_generated": parallel_runs[0].tokens
                },
                {
                    "run_id": 2,
                    "answer": parallel_runs[1].answer,
                    "confidence": parallel_runs[1].confidence,
                    "latency_seconds": parallel_runs[1].latency,
                    "tokens_generated": parallel_runs[1].tokens
                }
            ],
            voted_answer=voted_answer,
            execution={
                "total_latency_seconds": max(parallel_runs[0].latency, parallel_runs[1].latency),
                "total_tokens_generated": parallel_runs[0].tokens + parallel_runs[1].tokens,
                "compute_cost": self.calculate_compute_cost(parallel_runs)
            },
            send_to={
                "cluster_id": self.cluster_id,
                "super_cluster_id": self.super_cluster_id
            },
            timestamp=time.now()
        )
        
        # Step 7: Send result to cluster (if member) and super cluster
        if self.cluster_id:
            self.send_result_to_cluster(result)
        self.send_result_to_super_cluster(result)
        
        # Step 8: Update status
        self.runtime_state.status = "idle"
        self.runtime_state.current_task = None
        
        return result
    
    async def run_llm(self, prompt: str, run_id: int) -> LLMResponse:
        """
        ⚡ CRITICAL: Run a single LLM inference
        Each node runs this TWICE in parallel for voting
        """
        start_time = time.now()
        
        # Ensure model is loaded
        if not self.model_loaded:
            self.load_model()
        
        # Run inference using the Node's LLM (1B-13B params)
        response = self.llm_engine.generate(
            prompt=prompt,
            temperature=self.chromosome.sampling_params.temperature,
            top_p=self.chromosome.sampling_params.top_p,
            top_k=self.chromosome.sampling_params.top_k,
            max_tokens=self.chromosome.sampling_params.max_tokens,
            repetition_penalty=self.chromosome.sampling_params.repetition_penalty
        )
        
        # Calculate confidence (model's internal confidence score)
        confidence = self.calculate_confidence(response)
        
        return LLMResponse(
            answer=response.text,
            confidence=confidence,
            latency=time.now() - start_time,
            tokens=len(response.tokens),
            run_id=run_id
        )
    
    def vote_on_answers(self, run1: LLMResponse, run2: LLMResponse) -> VotedAnswer:
        """
        Compare two parallel LLM runs and vote on final answer
        USES ENSEMBLE WEIGHT from chromosome to weight votes
        """
        # Get ensemble weight for this node
        ensemble_weight = self.chromosome.ensemble_weight
        
        # Exact match - unanimous agreement
        if run1.answer.strip() == run2.answer.strip():
            return VotedAnswer(
                answer=run1.answer,
                vote_agreement="unanimous",
                confidence=(run1.confidence + run2.confidence) / 2,
                ensemble_weighted_confidence=(run1.confidence + run2.confidence) / 2 * ensemble_weight,
                reasoning=f"Both LLM runs produced identical answer (ensemble weight: {ensemble_weight})"
            )
        
        # Semantic similarity check (for near-matches)
        similarity = self.compute_semantic_similarity(run1.answer, run2.answer)
        
        if similarity > 0.95:  # Very similar answers
            # Use higher confidence answer, apply ensemble weight
            if run1.confidence >= run2.confidence:
                return VotedAnswer(
                    answer=run1.answer,
                    vote_agreement="majority",
                    confidence=run1.confidence,
                    ensemble_weighted_confidence=run1.confidence * ensemble_weight,
                    reasoning=f"Answers semantically similar, Run 1 had higher confidence (ensemble weight: {ensemble_weight})"
                )
            else:
                return VotedAnswer(
                    answer=run2.answer,
                    vote_agreement="majority",
                    confidence=run2.confidence,
                    ensemble_weighted_confidence=run2.confidence * ensemble_weight,
                    reasoning=f"Answers semantically similar, Run 2 had higher confidence (ensemble weight: {ensemble_weight})"
                )
        
        # Different answers - split vote, use confidence with ensemble weighting
        elif run1.confidence > run2.confidence:
            return VotedAnswer(
                answer=run1.answer,
                vote_agreement="split",
                confidence=run1.confidence,
                ensemble_weighted_confidence=run1.confidence * ensemble_weight,
                reasoning=f"Split vote - Run 1 had higher confidence ({run1.confidence:.2f} vs {run2.confidence:.2f}), ensemble weight: {ensemble_weight}"
            )
        else:
            return VotedAnswer(
                answer=run2.answer,
                vote_agreement="split",
                confidence=run2.confidence,
                ensemble_weighted_confidence=run2.confidence * ensemble_weight,
                reasoning=f"Split vote - Run 2 had higher confidence ({run2.confidence:.2f} vs {run1.confidence:.2f}), ensemble weight: {ensemble_weight}"
            )
```

### 2. Chromosome Management

```python
class ChromosomeManager:
    def apply_mutation(self, mutation: Mutation):
        """
        Apply parameter mutations (minor changes)
        """
        # Backup current chromosome
        self.chromosome_history.append(deepcopy(self.chromosome))
        old_hash = self.chromosome.genetic_hash
        
        # Apply parameter changes
        if mutation.parameter_changes:
            for param, new_value in mutation.parameter_changes.items():
                set_nested_value(self.chromosome.sampling_params, param, new_value)
        
        # Apply prompt changes
        if mutation.prompt_changes:
            self.chromosome.prompt_template.update(mutation.prompt_changes)
        
        # Compute new hash
        new_hash = compute_chromosome_hash(self.chromosome)
        self.chromosome.genetic_hash = new_hash
        
        # Update lineage
        self.chromosome.lineage.generation += 1
        self.chromosome.lineage.mutation_history.append({
            "generation": self.chromosome.lineage.generation,
            "type": "parameter_mutation",
            "old_hash": old_hash,
            "new_hash": new_hash,
            "changes": mutation
        })
        
        # Warm up model with new parameters
        self.warmup_model()
    
    def apply_type_specialization(self, old_type: str, new_type: str, reason: str):
        """
        Transform node from one type to another (MAJOR genetic change)
        Examples:
        - fast-generalist → math-specialist (discovers talent in math)
        - accurate-generalist → code-specialist (cluster needs more coders)
        """
        # Backup current chromosome
        self.chromosome_history.append(deepcopy(self.chromosome))
        old_hash = self.chromosome.genetic_hash
        
        # Update node type
        self.chromosome.node_type.primary_type = new_type
        self.node_identity.node_type = new_type  # Sync with identity
        
        # Load archetype for new type
        archetype = NODE_TYPE_ARCHETYPES[new_type]
        
        # Apply ENTIRE archetype (complete genetic overhaul)
        self.chromosome.sampling_params.temperature = archetype["temperature"]
        self.chromosome.prompt_template.system_prompt = archetype["system_prompt"]
        self.chromosome.fine_tuning.adapter = archetype["fine_tuning_adapter"]
        self.chromosome.specialization.primary_domain = archetype["primary_domain"]
        self.chromosome.prompt_template.chain_of_thought = archetype["chain_of_thought"]
        self.chromosome.sampling_params.max_tokens = archetype["max_tokens"]
        
        # Compute new hash (will be very different due to type change)
        new_hash = compute_chromosome_hash(self.chromosome)
        self.chromosome.genetic_hash = new_hash
        
        # Update lineage
        self.chromosome.lineage.generation += 1
        self.chromosome.lineage.type_evolution.append({
            "generation": self.chromosome.lineage.generation,
            "old_type": old_type,
            "new_type": new_type,
            "reason": reason
        })
        self.chromosome.lineage.mutation_history.append({
            "generation": self.chromosome.lineage.generation,
            "type": "type_specialization",
            "old_hash": old_hash,
            "new_hash": new_hash,
            "changes": {
                "node_type": f"{old_type} -> {new_type}",
                "all_parameters": "overwritten_by_archetype"
            }
        })
        
        # Reload model with new fine-tuning adapter
        self.reload_model_with_adapter(archetype["fine_tuning_adapter"])
        
        # Reset fitness (new chromosome, new evaluation needed)
        self.runtime_state.fitness.current_score = 0.5
        self.runtime_state.fitness.tasks_evaluated = 0
    
    def perform_crossover(self, other_chromosome: Chromosome, inheritance_map: dict):
        """
        Combine traits from two parent chromosomes
        Child inherits node_type from higher-fitness parent
        """
        new_chromosome = deepcopy(self.chromosome)
        old_hash = new_chromosome.genetic_hash
        
        # Determine child's node_type (from higher-fitness parent)
        if inheritance_map.child_type:
            new_chromosome.node_type.primary_type = inheritance_map.child_type
            self.node_identity.node_type = inheritance_map.child_type
        
        # Inherit specific parameters from other parent
        for trait_path in inheritance_map.from_other:
            value = get_nested_value(other_chromosome, trait_path)
            set_nested_value(new_chromosome, trait_path, value)
        
        # Compute new hash
        new_hash = compute_chromosome_hash(new_chromosome)
        new_chromosome.genetic_hash = new_hash
        
        # Update lineage
        new_chromosome.lineage.generation += 1
        new_chromosome.lineage.parent_chromosomes.append(other_chromosome.genetic_hash)
        new_chromosome.lineage.crossover_events.append({
            "generation": new_chromosome.lineage.generation,
            "parents": [self.chromosome.genetic_hash, other_chromosome.genetic_hash],
            "parent_types": [self.chromosome.node_type.primary_type, other_chromosome.node_type.primary_type],
            "child_type": new_chromosome.node_type.primary_type,
            "inheritance": inheritance_map
        })
        new_chromosome.lineage.mutation_history.append({
            "generation": new_chromosome.lineage.generation,
            "type": "crossover",
            "old_hash": old_hash,
            "new_hash": new_hash,
            "changes": inheritance_map
        })
        
        # Replace chromosome
        self.chromosome = new_chromosome
        self.runtime_state.active_chromosome = new_hash
        
        # Reset fitness
        self.runtime_state.fitness.current_score = 0.5
        self.runtime_state.fitness.tasks_evaluated = 0
        
        # Reload model
        self.reload_model()
    
    def rollback_chromosome(self, generations: int = 1):
        """
        Revert to previous chromosome state (if update failed)
        This reverts node_type as well if it changed
        """
        if len(self.chromosome_history) < generations:
            raise ValueError("Not enough chromosome history")
        
        # Restore previous chromosome
        self.chromosome = self.chromosome_history[-(generations)]
        self.chromosome_history = self.chromosome_history[:-(generations)]
        
        # Sync node_type with restored chromosome
        self.node_identity.node_type = self.chromosome.node_type.primary_type
        
        # Reload model with previous parameters
        self.reload_model()
```

### 3. Resource Monitoring

```python
class ResourceMonitor:
    def monitor_resources(self):
        """
        Continuously monitor resource usage
        """
        while True:
            # CPU monitoring
            self.runtime_state.resources.cpu_usage = psutil.cpu_percent()
            
            # Memory monitoring
            memory = psutil.virtual_memory()
            self.runtime_state.resources.memory_usage = memory.used / (1024**3)  # GB
            
            # GPU monitoring (if available)
            if self.has_gpu:
                gpu_stats = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                self.runtime_state.resources.gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
                self.runtime_state.resources.gpu_memory = gpu_stats.used / (1024**3)
            
            # Battery monitoring (for mobile)
            if self.is_mobile:
                battery = psutil.sensors_battery()
                self.runtime_state.resources.battery_level = battery.percent
            
            # Thermal monitoring
            temps = psutil.sensors_temperatures()
            avg_temp = sum(t.current for t in temps.values()) / len(temps)
            if avg_temp > 80:
                self.runtime_state.resources.thermal_state = "hot"
            elif avg_temp > 65:
                self.runtime_state.resources.thermal_state = "warm"
            else:
                self.runtime_state.resources.thermal_state = "normal"
            
            # Adjust task acceptance based on resources
            self.adjust_task_acceptance()
            
            time.sleep(5)  # Check every 5 seconds
    
    def adjust_task_acceptance(self):
        """
        Dynamically adjust willingness to accept tasks based on resources
        """
        # Reduce task acceptance if overheating
        if self.runtime_state.resources.thermal_state == "hot":
            self.max_queue_size = 1
            self.reject_new_tasks = True
        
        # Reduce task acceptance if low battery (mobile)
        if self.is_mobile and self.runtime_state.resources.battery_level < 20:
            self.max_queue_size = 0
            self.reject_new_tasks = True
        
        # Reduce task acceptance if high CPU usage
        if self.runtime_state.resources.cpu_usage > 90:
            self.max_queue_size = 2
```

### 4. Credit Management

```python
class CreditManager:
    def calculate_credits_earned(self, task_result: TaskResult, feedback: PerformanceFeedback) -> int:
        """
        Calculate credits earned for a completed task
        """
        # Base rate (credits per task)
        base_rate = 30
        
        # Quality multiplier (based on correctness)
        quality_multiplier = 1.0 + (feedback.outcome.correctness - 0.5)  # 0.5 to 1.5
        
        # Uptime bonus (reliable nodes earn more)
        uptime_hours = self.runtime_state.performance.uptime_hours
        if uptime_hours > 1000:
            uptime_bonus = 1.2
        elif uptime_hours > 500:
            uptime_bonus = 1.1
        else:
            uptime_bonus = 1.0
        
        # Difficulty factor (harder tasks earn more)
        difficulty_map = {"simple": 1.0, "medium": 1.1, "complex": 1.3}
        difficulty_factor = difficulty_map.get(task_result.task_complexity, 1.0)
        
        # Model size factor (larger models earn more per task)
        model_size_gb = self.genome.base_model.size_gb
        if model_size_gb >= 10:  # 13B model
            model_factor = 1.3
        elif model_size_gb >= 5:  # 7B model
            model_factor = 1.2
        else:  # 1-3B model
            model_factor = 1.0
        
        # Calculate total
        credits = int(
            base_rate * 
            quality_multiplier * 
            uptime_bonus * 
            difficulty_factor * 
            model_factor
        )
        
        return credits
    
    def update_credit_balance(self, transaction: CreditTransaction):
        """
        Update local credit balance
        """
        if transaction.type == "earn":
            self.runtime_state.credits.balance += transaction.earn.amount
            self.runtime_state.credits.earned_lifetime += transaction.earn.amount
        
        elif transaction.type == "spend":
            self.runtime_state.credits.balance -= transaction.spend.amount
            self.runtime_state.credits.spent_lifetime += transaction.spend.amount
        
        elif transaction.type == "bonus":
            self.runtime_state.credits.balance += transaction.bonus.amount
        
        elif transaction.type == "penalty":
            self.runtime_state.credits.balance -= transaction.penalty.amount
        
        # Update earning rate (credits per hour)
        self.calculate_earning_rate()
```

### 5. Self-Assessment & Quality Control

```python
class QualityController:
    def self_assess_quality(self, result: ParsedResult, task: TaskAssignment) -> QualityIndicators:
        """
        Node performs self-assessment before sending result
        """
        indicators = {
            "passed_self_check": True,
            "internal_consistency": True,
            "format_valid": True,
            "within_constraints": True
        }
        
        # Check 1: Format validation
        if task.context.format_requirements:
            if not self.validate_format(result.answer, task.context.format_requirements):
                indicators["format_valid"] = False
                indicators["passed_self_check"] = False
        
        # Check 2: Constraint adherence
        if len(result.answer.split()) > task.constraints.max_tokens:
            indicators["within_constraints"] = False
            indicators["passed_self_check"] = False
        
        # Check 3: Internal consistency (for math/logic)
        if task.task_type in ["calculation", "logical_reasoning"]:
            if not self.check_internal_consistency(result):
                indicators["internal_consistency"] = False
                indicators["passed_self_check"] = False
        
        # Check 4: Confidence threshold
        if result.confidence < task.constraints.min_confidence:
            indicators["passed_self_check"] = False
        
        return QualityIndicators(**indicators)
    
    def check_internal_consistency(self, result: ParsedResult) -> bool:
        """
        Verify logical consistency in the answer
        """
        # For mathematical answers, verify calculation steps
        if result.reasoning and result.reasoning.intermediate_steps:
            for i, step in enumerate(result.reasoning.intermediate_steps[:-1]):
                if not self.step_follows_from_previous(step, result.reasoning.intermediate_steps[i+1]):
                    return False
        
        # For code, run syntax check
        if result.format == "code":
            try:
                compile(result.answer, '<string>', 'exec')
            except SyntaxError:
                return False
        
        return True
```

---

## Node Lifecycle: Spawning, Killing, State Changes

### Node Spawning Events

A node is spawned in three scenarios:

#### 1. New User Joins Network

```python
def spawn_node_for_new_user(user_id: str, device_specs: dict) -> Node:
    """
    When a new user joins DELLM, spawn their first node
    """
    # Determine device tier
    device_tier = detect_device_tier(device_specs)
    
    # Super Cluster decides initial chromosome based on network needs
    initial_chromosome = super_cluster.allocate_genesis_chromosome(
        device_tier=device_tier,
        network_needs={
            "math_specialists_needed": 15,
            "code_specialists_needed": 8,
            "generalists_needed": 50
        }
    )
    
    # Create new node
    node = Node(
        node_id=generate_uuid(),
        user_id=user_id,
        super_cluster_id="super-cluster-main",
        cluster_id=None,  # Will be assigned based on specialization
        chromosome=initial_chromosome,
        spawned_by="new_user",
        parent_node_id=None,
        fitness=FitnessState(
            current_score=0.5,  # Start at neutral fitness
            tasks_evaluated=0
        )
    )
    
    # Super Cluster assigns to appropriate cluster
    cluster_id = super_cluster.assign_to_cluster(
        node=node,
        specialization=initial_chromosome.specialization.primary_domain
    )
    node.cluster_id = cluster_id
    
    # Give starter credits
    node.runtime_state.credits_contributed.balance = 100  # Welcome bonus
    
    return node
```

#### 2. Evolutionary Replacement (Kill + Spawn)

```python
def spawn_node_from_evolution(
    parent_node: Node,
    mutation_type: str,
    cluster_id: str
) -> Node:
    """
    When a node is killed in evolution, spawn a new one from a successful parent
    """
    # Get parent's chromosome
    parent_chromosome = parent_node.chromosome
    
    # Apply mutation to create child chromosome
    child_chromosome = apply_mutation(
        parent=parent_chromosome,
        mutation_type=mutation_type  # "parameter_tweak" | "prompt_variation" | "crossover"
    )
    
    # Create new node (inherits parent's user_id)
    node = Node(
        node_id=generate_uuid(),
        user_id=parent_node.user_id,  # Same user owns the evolved node
        super_cluster_id=parent_node.super_cluster_id,
        cluster_id=cluster_id,
        chromosome=child_chromosome,
        spawned_by="evolution",
        parent_node_id=parent_node.node_id,
        fitness=FitnessState(
            current_score=0.5,  # Start at neutral, will improve or decline
            tasks_evaluated=0
        )
    )
    
    return node
```

#### 3. Cluster/Super Cluster Initiated Spawning

```python
def spawn_node_for_specialization(
    user_id: str,
    specialization_needed: str,
    cluster_id: str,
    super_cluster_id: str
) -> Node:
    """
    Cluster or Super Cluster decides to spawn a specialized node
    """
    # Create chromosome optimized for needed specialization
    specialized_chromosome = create_specialized_chromosome(
        specialization=specialization_needed,  # e.g., "mathematical_reasoning"
        base_model_tier=get_user_device_tier(user_id)
    )
    
    # Create new node
    node = Node(
        node_id=generate_uuid(),
        user_id=user_id,
        super_cluster_id=super_cluster_id,
        cluster_id=cluster_id,
        chromosome=specialized_chromosome,
        spawned_by="cluster_request",
        parent_node_id=None,
        fitness=FitnessState(
            current_score=0.5,
            tasks_evaluated=0
        )
    )
    
    return node
```

### Node Killing (Death)

A node is killed during evolutionary culling:

```python
def kill_node(node: Node, reason: str):
    """
    Kill a node during evolutionary process
    
    Triggered by: Cluster or Super Cluster evolutionary culling
    Reason: Low fitness score (bottom 20% of population)
    """
    # Verify node is eligible for culling
    if node.fitness.tasks_evaluated < MIN_TASKS_FOR_EVALUATION:
        # Don't kill nodes that haven't been evaluated enough
        return False
    
    if node.fitness.current_score < SURVIVAL_THRESHOLD:
        # Mark node as killed
        node.runtime_state.status = "killed"
        
        # Transfer remaining credits back to user pool
        user_credit_balance[node.user_id] += node.runtime_state.credits_contributed.balance
        
        # Notify cluster/super cluster
        if node.cluster_id:
            cluster.notify_node_death(node.node_id)
        super_cluster.notify_node_death(node.node_id)
        
        # Log death for evolutionary tracking
        evolutionary_log.append({
            "event": "node_death",
            "node_id": node.node_id,
            "chromosome": node.chromosome.genetic_hash,
            "final_fitness": node.fitness.current_score,
            "tasks_evaluated": node.fitness.tasks_evaluated,
            "reason": reason,
            "timestamp": time.now()
        })
        
        # Unload model and free resources
        node.unload_model()
        
        # Remove from cluster/super cluster
        node.disconnect_from_network()
        
        return True
    
    return False
```

### State Transitions

The node's state changes in response to specific events:

#### State Diagram

```
[New User] 
    ↓ (spawn_node_for_new_user)
[Idle] ←→ [Executing] ←→ [Mutating]
    ↓                        ↑
[Killed] ← (fitness < threshold)
    
[Idle/Executing/Mutating]
    ↓ (fitness update)
[Fitness Updated] → [Idle/Executing/Mutating]
```

#### State Change Events

```python
class NodeStateMachine:
    
    def on_spawn(self, spawn_type: str):
        """
        State: None → Idle
        Event: Node is spawned
        """
        self.runtime_state.status = "idle"
        self.load_model()
        self.connect_to_network()
        self.broadcast_availability()
    
    def on_task_received(self, task: TaskAssignment):
        """
        State: Idle → Executing
        Event: Task assignment received
        """
        if self.runtime_state.status == "idle":
            self.runtime_state.status = "executing"
            self.runtime_state.current_task = task.task_id
            # Execute task (run 2 LLMs in parallel)
    
    def on_task_completed(self):
        """
        State: Executing → Idle
        Event: Task execution finished
        """
        if self.runtime_state.status == "executing":
            self.runtime_state.status = "idle"
            self.runtime_state.current_task = None
            self.broadcast_availability()
    
    def on_fitness_update(self, fitness_update: FitnessUpdate):
        """
        State: Any → Same (but fitness score changed)
        Event: Cluster/Super Cluster verifies answer and updates fitness
        """
        # Update fitness score (weighted average)
        self.apply_fitness_update(fitness_update)
        
        # Chromosome hash does NOT change (same genetic configuration)
        # Only the fitness score associated with this chromosome changes
        
        # Check if fitness dropped below survival threshold
        if self.runtime_state.fitness.current_score < SURVIVAL_THRESHOLD:
            # Flag for potential culling in next evolutionary cycle
            self.flagged_for_culling = True
    
    def on_mutation_triggered(self, mutation: Mutation):
        """
        State: Idle → Mutating → Idle
        Event: Cluster triggers evolutionary mutation
        """
        if self.runtime_state.status == "idle":
            self.runtime_state.status = "mutating"
            
            # Apply mutation to chromosome
            old_chromosome_hash = self.chromosome.genetic_hash
            self.apply_mutation(mutation)
            new_chromosome_hash = self.chromosome.genetic_hash
            
            # Update active chromosome reference
            self.runtime_state.active_chromosome = new_chromosome_hash
            
            # Chromosome has changed - genetic hash is different
            # Fitness score resets to neutral for new chromosome
            self.runtime_state.fitness.current_score = 0.5
            self.runtime_state.fitness.tasks_evaluated = 0
            
            # Reload model with new parameters
            self.reload_model()
            
            self.runtime_state.status = "idle"
            self.broadcast_availability()
    
    def on_culling_triggered(self):
        """
        State: Any → Killed
        Event: Cluster/Super Cluster culls low-fitness nodes
        """
        if self.fitness.current_score < SURVIVAL_THRESHOLD:
            self.runtime_state.status = "killed"
            kill_node(self, reason="low_fitness")
```

### Chromosome State Changes

**CRITICAL: Chromosome hash changes when genetic configuration changes**

```python
def chromosome_state_changes():
    """
    Chromosome (genetic_hash) changes ONLY when:
    1. Mutation occurs (parameter tweak, prompt change)
    2. Crossover occurs (inheriting genes from parents)
    3. Fine-tuning applied (new LoRA adapter)
    
    Chromosome does NOT change when:
    1. Fitness score updates (same genes, different performance)
    2. Task execution (same genes, different task)
    3. Credits earned (same genes, different economics)
    """
    
    # Example: Mutation changes chromosome
    old_chromosome = "chr-a3f5e8d9c2b1"
    apply_mutation(temperature: 0.3 -> 0.35)
    new_chromosome = "chr-b7k2m9f4e8x3"  # Different hash
    
    # Example: Fitness update does NOT change chromosome
    chromosome = "chr-a3f5e8d9c2b1"
    update_fitness(0.82 -> 0.85)
    chromosome_after = "chr-a3f5e8d9c2b1"  # Same hash
```

### Fitness Score Updates

**Fitness score changes when Cluster/Super Cluster verifies answer:**

```python
def update_fitness_score(node: Node, verification: VerificationResult):
    """
    Fitness formula: weighted_correctness / (1 + latency_normalized)
    
    Updated using exponential moving average to smooth out variance
    """
    # Calculate task fitness
    task_fitness = verification.correctness / (1 + verification.latency_normalized)
    
    # Update node fitness (weighted average)
    # Old fitness has 90% weight, new task has 10% weight
    old_fitness = node.runtime_state.fitness.current_score
    new_fitness = 0.9 * old_fitness + 0.1 * task_fitness
    
    # Update state
    node.runtime_state.fitness.current_score = new_fitness
    node.runtime_state.fitness.tasks_evaluated += 1
    node.runtime_state.fitness.last_updated = time.now()
    
    # Add to history
    node.runtime_state.fitness.score_history.append({
        "timestamp": time.now(),
        "score": new_fitness,
        "task_id": verification.task_id
    })
```

### Summary: Node State Changes

| Event | Old State | New State | Node Type Changes? | Chromosome Changes? | Fitness Changes? |
|-------|-----------|-----------|-------------------|---------------------|------------------|
| **Spawn (new user)** | None | Idle | N/A (assigned type) | N/A (new chromosome) | N/A (starts at 0.5) |
| **Task received** | Idle | Executing | No | No | No |
| **Task completed** | Executing | Idle | No | No | No |
| **Fitness verified** | Any | Same | No | No | Yes (weighted avg) |
| **Parameter mutation** | Idle | Mutating → Idle | No | Yes (new hash) | Yes (resets to 0.5) |
| **Type specialization** | Idle | Mutating → Idle | **Yes** (e.g., generalist → specialist) | **Yes** (major hash change) | Yes (resets to 0.5) |
| **Type shift** | Idle | Mutating → Idle | **Yes** (e.g., math → code specialist) | **Yes** (major hash change) | Yes (resets to 0.5) |
| **Crossover** | None | Idle | **Maybe** (inherits from parent) | Yes (new hash) | Yes (starts at 0.5) |
| **Culled** | Any | Killed | No | No | No (already low) |

### Chromosome Hash Changes

**When does the chromosome hash change?**

✅ **YES - Hash changes:**
- Parameter mutation (temperature, top_p, etc.)
- Prompt template modification
- **Node type specialization** (generalist → specialist)
- **Node type shift** (specialist → different specialist)
- Fine-tuning adapter update
- Crossover (combining parent genes)

❌ **NO - Hash stays same:**
- Fitness score updates
- Task execution
- Credit balance changes
- Network connectivity changes
- Resource usage changes

**Critical: node_type is PART of the chromosome hash**
```python
# Same parameters but different type = different chromosome
chr-math-a3f5e8d9c2b1  # math-specialist, temp=0.2
chr-code-a3f5e8d9c2b1  # code-specialist, temp=0.2 (different hash!)

# Type changes ALWAYS create new chromosome hash
chr-fgen-123abc  # fast-generalist
    ↓ type specialization
chr-math-789xyz  # math-specialist (completely different hash)
```

---

## Hardware Adaptation

### Device Tier Detection

```python
class HardwareAdapter:
    def detect_device_tier(self) -> str:
        """
        Determine device capabilities and select appropriate lightweight model
        
        CRITICAL: All tiers use ultra-lightweight models (0.6-2.3 GB)
        Never use 7B+ models - reserved for Super LLM!
        """
        # Get hardware specs
        cpu_cores = psutil.cpu_count()
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        has_gpu = self.detect_gpu()
        
        # High-tier devices (Gaming PCs, Workstations)
        if has_gpu:
            gpu_memory_gb = self.get_gpu_memory_gb()
            if gpu_memory_gb >= 8 and total_ram_gb >= 16:
                return "high-tier"  # Can run 3.8B models
            else:
                return "mid-tier"   # Can run 2.7B models
        
        # Laptops and desktops
        elif total_ram_gb >= 8 and cpu_cores >= 4:
            return "mid-tier"   # Can run 2.7B models
        
        # Budget laptops, tablets
        elif total_ram_gb >= 4:
            return "low-tier"   # Can run 1.6B models
        
        # Smartphones, old devices
        else:
            return "mobile"     # Can run 1.1B models
    
    def select_optimal_model(self, tier: str) -> ModelConfig:
        """
        Choose lightweight model based on hardware tier
        
        Strategy: Use SMALLEST possible model for maximum participation
        Ensemble voting compensates for individual model weakness
        """
        model_configs = {
            "high-tier": {
                "name": "phi-3-mini-3.8b",  # OPTIONAL upgrade for gaming PCs
                "size_gb": 2.3,
                "quantization": "Q4_K_M",
                "context_window": 4096,
                "ram_required_gb": 6,
                "inference_speed_tokens_sec": 30,
                "note": "Optional - can also use mid-tier model"
            },
            "mid-tier": {
                "name": "phi-2-2.7b",  # For laptops/desktops
                "size_gb": 1.6,
                "quantization": "Q4_K_M",
                "context_window": 2048,
                "ram_required_gb": 4,
                "inference_speed_tokens_sec": 35,
                "note": "Good balance of quality and accessibility"
            },
            "low-tier": {
                "name": "stablelm-zephyr-1.6b",  # For budget devices
                "size_gb": 0.9,
                "quantization": "Q4_K_M",
                "context_window": 2048,
                "ram_required_gb": 2.5,
                "inference_speed_tokens_sec": 40,
                "note": "Works on most devices from 2018+"
            },
            "mobile": {
                "name": "tinyllama-1.1b-chat",  # DEFAULT - max participation
                "size_gb": 0.6,
                "quantization": "Q4_K_M",
                "context_window": 2048,
                "ram_required_gb": 1.5,
                "inference_speed_tokens_sec": 50,
                "note": "Runs on ANY modern smartphone"
            }
        }
        
        return ModelConfig(**model_configs[tier])
    
    def get_device_examples(self, tier: str) -> list:
        """
        Example devices for each tier
        """
        examples = {
            "high-tier": [
                "Desktop with RTX 3060+",
                "Gaming PC with 16GB+ RAM",
                "Workstation with dedicated GPU"
            ],
            "mid-tier": [
                "MacBook Air M1/M2/M3",
                "Windows laptop (2020+)",
                "Desktop PC (8GB+ RAM)"
            ],
            "low-tier": [
                "Budget laptop (4-8GB RAM)",
                "Chromebook",
                "iPad Pro",
                "Older laptops (2018+)"
            ],
            "mobile": [
                "iPhone 11 or newer",
                "Android flagship (2020+)",
                "Mid-range smartphone",
                "Tablet"
            ]
        }
        return examples[tier]
```

---

## Persistence & Recovery

### State Persistence

```python
class StatePersistence:
    def save_state(self):
        """
        Periodically save node state to disk for recovery
        """
        state_snapshot = {
            "node_id": self.node_id,
            "genome": self.genome,
            "runtime_state": self.runtime_state,
            "genome_history": self.genome_history[-10:],  # Last 10 genomes
            "performance_history": self.performance_history[-1000:],  # Last 1000 tasks
            "credit_balance": self.runtime_state.credits.balance,
            "timestamp": time.now()
        }
        
        # Save to disk
        with open(f"/var/dellm/nodes/{self.node_id}/state.json", "w") as f:
            json.dump(state_snapshot, f)
    
    def recover_state(self):
        """
        Recover node state after crash or restart
        """
        state_file = f"/var/dellm/nodes/{self.node_id}/state.json"
        
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state_snapshot = json.load(f)
            
            # Restore genome
            self.genome = Genome(**state_snapshot["genome"])
            
            # Restore runtime state
            self.runtime_state = RuntimeState(**state_snapshot["runtime_state"])
            
            # Restore history
            self.genome_history = state_snapshot["genome_history"]
            self.performance_history = state_snapshot["performance_history"]
            
            # Mark as recovered
            self.runtime_state.status = "recovered"
        else:
            # Fresh start - initialize with default genesis genome
            self.initialize_genesis_genome()
```

---

## Summary

The Node Block is the fundamental execution unit of DELLM, containing:

**Core Components:**
- **LLM Instance (1B-13B params)** - The actual language model
- **Genome** - Evolutionary DNA that defines behavior
- **Runtime State** - Current execution and resource status
- **Credit Manager** - Economic incentive tracking

**Key Inputs:**
1. Task assignments from Super Cluster/Cluster
2. Evolutionary updates from Cluster
3. Performance feedback and fitness signals
4. Credit transactions

**Key Outputs:**
1. Task results to Super Cluster/Cluster
2. Performance reports and metrics
3. Availability broadcasts
4. Evolution feedback

**Internal Mechanisms:**
- Task execution pipeline with LLM inference
- Genome management and evolution
- Resource monitoring and adaptation
- Quality self-assessment
- State persistence and recovery

The Node Block adapts to different hardware tiers, participates in evolutionary improvement, and earns credits through quality contributions to the network.

## Evolutionary Mutation Strategy

### Mutation Type Distribution

**CRITICAL PRINCIPLE: Focus on node_type and ensemble_weight, minimize LLM parameter changes**

```python
MUTATION_DISTRIBUTION = {
    "type_shift": 0.50,          # 50% - Change node type (math → code, generalist → specialist)
    "weight_adjustment": 0.30,    # 30% - Adjust ensemble voting weight
    "type_refinement": 0.15,      # 15% - Improve type config (prompts, examples)
    "parameter_tweak": 0.05       # 5% - RARE: Tweak LLM sampling params (risky!)
}
```

### Why This Distribution?

1. **Type shifts (50%)** - Most impactful changes
   - Moving from generalist → specialist can 2x fitness
   - Adapts node to network demands
   - Predictable outcomes (we know what each type does)

2. **Weight adjustments (30%)** - Fine-tune trust
   - High-fitness nodes get more voting power
   - Low-fitness nodes get less voting power
   - Gradual optimization without risk

3. **Type refinement (15%)** - Incremental improvement
   - Better few-shot examples from successful tasks
   - Improved prompts based on experience
   - Low risk, moderate reward

4. **Parameter tweaks (5%)** - Exploration only
   - Unpredictable results
   - Could improve OR hurt performance
   - Only for genetic diversity

### Example Mutation Scenarios

**Scenario 1: Generalist → Math Specialist**
```python
# Before mutation
chromosome = {
    "node_type": "generalist",
    "ensemble_weight": 1.0,
    "type_config": {...general config...}
}

# Cluster observes: node excels at math tasks (0.92 correctness)
# Cluster triggers: type_shift mutation

# After mutation  
chromosome = {
    "node_type": "math_specialist",  # ← CHANGED
    "ensemble_weight": 1.0,
    "type_config": {  # ← COMPLETELY REPLACED
        "system_prompt": "You are a math expert...",
        "few_shot_examples": [...math examples...],
        "fine_tuning_adapter": "lora-math-v2"  # ← CHANGED
    }
}

# Result: Fitness improves from 0.75 → 0.85 on math tasks
```

**Scenario 2: Weight Increase for High Performer**
```python
# Node has been consistently correct (fitness trending up)
# Cluster triggers: weight_adjustment mutation

# Before
ensemble_weight = 1.0

# After
ensemble_weight = 1.3

# Result: This node's votes now count for 1.3× in ensemble decisions
# Other nodes' votes are weighted accordingly
```

**Scenario 3: Type Refinement from Experience**
```python
# Node (code_specialist) has completed 100+ tasks
# Extract best examples from successful tasks
# Trigger: type_refinement mutation

# Before
type_config.few_shot_examples = [
    {"query": "Write a loop", "answer": "for i in range(10): ..."}
]

# After (learned from successful tasks)
type_config.few_shot_examples = [
    {"query": "Write a loop", "answer": "for i in range(10): ..."},
    {"query": "Handle exceptions", "answer": "try:\n    ...\nexcept ValueError:\n    ..."},
    {"query": "List comprehension", "answer": "[x**2 for x in range(10)]"}
]

# Result: Marginal fitness improvement (0.82 → 0.84)
```

**Scenario 4: Risky Parameter Tweak (rare)**
```python
# 5% exploration mutation
# Cluster triggers: parameter_tweak

# Before
sampling_params.temperature = 0.7

# After
sampling_params.temperature = 0.6

# Result: UNPREDICTABLE
# Could improve precision (+0.05 fitness)
# Could hurt creativity (-0.03 fitness)
# Monitored closely, rolled back if fitness drops
```


---

## Why Ultra-Lightweight Models Work in DELLM

### The Ensemble Advantage

**Key Insight:** Individual node quality matters less than ensemble quality

```python
# Example: Math task "What is 15% of 240?"

# Single large model (7B)
single_7b_node = {
    "correctness": 0.95,
    "confidence": 0.90,
    "latency": 1.2
}

# Ensemble of 5 tiny models (1.1B each)
ensemble_5x_tiny = {
    "node_1": {"answer": "36", "confidence": 0.82},
    "node_2": {"answer": "36", "confidence": 0.85},
    "node_3": {"answer": "36", "confidence": 0.79},
    "node_4": {"answer": "36", "confidence": 0.88},
    "node_5": {"answer": "35.9", "confidence": 0.75},
    
    # Voting result
    "ensemble_vote": "36",  # 4/5 agree
    "combined_correctness": 0.98,  # Higher than single 7B!
    "average_latency": 0.8,  # Faster (parallel execution)
}
```

**Result:** 5 tiny models > 1 large model, at lower individual cost!

### Network Effects

```
Lightweight Models (1.1B):
├─ Can run on: 1+ billion smartphones
├─ Barrier to entry: Very low
├─ Network growth: Fast (everyone can participate)
├─ Total network capacity: Massive
└─ Ensemble quality: High (many voters)

Heavy Models (7B):
├─ Can run on: ~100 million gaming PCs
├─ Barrier to entry: High (need good hardware)
├─ Network growth: Slow (limited devices)
├─ Total network capacity: Limited
└─ Ensemble quality: Lower (fewer voters)
```

### Economic Advantages

```python
TinyLlama 1.1B Node:
├─ Power consumption: ~2W
├─ Inference speed: 50 tokens/sec
├─ Credits earned per hour: 40
├─ Credits per watt: 20
└─ Can run 24/7 on phone

Llama 7B Node:
├─ Power consumption: ~50W
├─ Inference speed: 10 tokens/sec
├─ Credits earned per hour: 60 (better quality)
├─ Credits per watt: 1.2
└─ Needs gaming PC, can't run 24/7

Winner: Tiny model has 16× better credit/watt efficiency!
```

### Division of Labor

**DELLM's architecture naturally divides intelligence:**

```
┌─────────────────────────────────────────┐
│ Super LLM (70B+)                        │
│ - Complex reasoning                     │
│ - Query decomposition                   │
│ - Final synthesis                       │
│ - Runs on: Dedicated infrastructure     │
└─────────────────────────────────────────┘
              ↓ (decomposes into simple subtasks)
┌─────────────────────────────────────────┐
│ Node LLMs (1.1B-3.8B)                   │
│ - Simple subtask execution              │
│ - Basic calculations                    │
│ - Fact retrieval                        │
│ - Runs on: Consumer devices             │
└─────────────────────────────────────────┘
```

**Key Point:** Nodes don't need to be smart - they just need to handle simple tasks correctly. The Super LLM does the hard thinking!

### Quality Comparison

```
Task: "Calculate compound interest on $10,000 at 5% for 10 years"

Super LLM (70B):
└─ Decomposes into:
    ├─ Subtask 1: "What is the compound interest formula?"
    ├─ Subtask 2: "Calculate (1.05)^10"
    └─ Subtask 3: "Calculate $10,000 × result - $10,000"

Each subtask is SIMPLE enough for 1.1B model to handle!

Node A (TinyLlama 1.1B):
├─ Subtask 2: "(1.05)^10 = 1.6289" ✓ Correct
└─ Confidence: 0.85

Node B (TinyLlama 1.1B):
├─ Subtask 2: "(1.05)^10 = 1.6289" ✓ Correct
└─ Confidence: 0.88

Node C (TinyLlama 1.1B):
├─ Subtask 2: "(1.05)^10 = 1.629" ✓ Close enough
└─ Confidence: 0.79

Ensemble vote: "1.6289" (unanimous) ✓
Combined confidence: 0.97

Super LLM synthesizes:
"$10,000 × 1.6289 = $16,289. Subtract principal: $16,289 - $10,000 = $6,289 interest."
```

### Specialization Still Works

Even with tiny models, node types matter:

```python
TinyLlama-1.1B as math_specialist:
├─ Fine-tuned on math problems
├─ System prompt: "You are a math expert..."
├─ Few-shot examples: Math-specific
├─ Performance on math: 0.85
└─ Performance on creative: 0.60

TinyLlama-1.1B as creative_specialist:
├─ Fine-tuned on creative writing
├─ System prompt: "You are a creative writer..."
├─ Few-shot examples: Creative-specific
├─ Performance on creative: 0.82
└─ Performance on math: 0.65

Same base model, different specialization = different strengths!
```

### Verification is Easier

With lightweight models, we can afford more redundancy:

```python
# Heavy model approach (7B)
redundancy = 2  # Too expensive to run 5×
verification_cost = 2 × 50W = 100W

# Lightweight model approach (1.1B)
redundancy = 5  # Cheap enough for 5× redundancy
verification_cost = 5 × 2W = 10W

Result: 10× cheaper verification, 2.5× more voting nodes!
```

### Real-World Performance

**Expected accuracy on common tasks:**

| Task Type | Single 1.1B | Ensemble 5× 1.1B | Single 7B |
|-----------|-------------|------------------|-----------|
| Simple math | 0.80 | 0.95 | 0.92 |
| Basic code | 0.75 | 0.92 | 0.89 |
| Factual Q&A | 0.85 | 0.96 | 0.94 |
| Simple logic | 0.78 | 0.93 | 0.90 |

**Key finding:** Ensemble of tiny models matches or exceeds single large model!

### Latency Benefits

```python
# Sequential processing (traditional)
7B_model_latency = 1.2 seconds

# Parallel processing (DELLM)
parallel_latency = max([
    tiny_model_1: 0.8s,
    tiny_model_2: 0.7s,
    tiny_model_3: 0.9s,
    tiny_model_4: 0.8s,
    tiny_model_5: 0.7s
]) = 0.9 seconds

DELLM is FASTER despite using smaller models!
```

### Bottom Line

**Lightweight models (1.1B-3.8B) are OPTIMAL for DELLM because:**

1. ✅ **Ensemble compensates** - 5 tiny > 1 large
2. ✅ **More participation** - 10× more devices can join
3. ✅ **Better economics** - 16× better credits/watt
4. ✅ **Faster network** - 5× faster inference
5. ✅ **Lower barrier** - Anyone with a phone can participate
6. ✅ **Division of labor** - Super LLM handles complex reasoning
7. ✅ **More redundancy** - Cheaper to run 5× verification
8. ✅ **Network effects** - Larger network = better overall quality

**Never use 7B+ for nodes - that's what the centralized Super LLM is for!**

