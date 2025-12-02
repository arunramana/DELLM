# Cluster Block: Detailed Architecture Specification

## Overview

The Cluster Block represents a "symbiotic group" in the DELLM ecosystem - a collection of Node "species" working together, sharing specializations, and evolving as a coordinated unit. It sits between the Super Cluster (ecosystem orchestrator) and individual Nodes (species).

**CRITICAL REMINDER:** The Cluster Block does NOT contain an LLM. It is pure coordination logic that manages groups of Nodes, routes tasks internally, performs verification, and triggers evolutionary changes.

**Biological Metaphor:**
- **Node** = Individual species (single organism)
- **Cluster** = Symbiotic group (organisms working together, like a coral reef or mycorrhizal network)
- **Super Cluster** = Ecosystem (global orchestrator)

---

## Cluster Identity & State

### Unique Identifier & Membership

```python
ClusterIdentity = {
    # Unique Cluster ID
    cluster_id: "cluster-math-specialists-001",
    
    # Network Position
    super_cluster_id: "super-cluster-main",  # Which super cluster this cluster belongs to
    
    # Cluster Type - Reflects the specialization
    cluster_type: "math_specialist" | "code_specialist" | "creative_specialist" | 
                  "analytical_specialist" | "factual_specialist" | 
                  "conversational_specialist" | "generalist" | "mixed",
    
    # Member Nodes
    member_nodes: [
        "node-550e8400-e29b-41d4-a716-446655440000",
        "node-661f9511-f30c-52e5-b827-557766551111",
        "node-772g0622-g41d-63f6-c938-668877662222",
        # ... up to N nodes
    ],
    
    # Population Stats
    population: {
        current_size: 45,
        min_size: 10,          # Cluster dissolves below this
        max_size: 100,         # Cluster splits above this
        target_size: 50,       # Optimal size
        growth_rate: 0.05      # 5% growth per generation
    },
    
    # Creation Metadata
    created_at: timestamp,
    spawned_by: "super_cluster_init" | "cluster_split" | "cluster_merge",
    parent_cluster_id: "cluster-parent-123" | null,  # Null if initial cluster
    generation: 7,  # How many evolutionary cycles since creation
    
    # Geographic/Network Info
    geographic_region: "us-west" | "eu-central" | "asia-east" | "global",
    average_latency_to_super_cluster_ms: 35
}
```

### Cluster Genome (Collective Genetic Profile)

**The cluster genome represents the collective genetic characteristics of all member nodes**

```python
ClusterGenome = {
    # Genetic Hash - Signature of cluster's collective genetic state
    genetic_hash: "clstr-math-a7f9e2d5c8b3",
    generation: 7,
    
    # ========================================
    # CLUSTER TYPE (PRIMARY IDENTIFIER)
    # ========================================
    
    cluster_type: "math_specialist",
    # Available types match node types:
    # - "math_specialist": Cluster specializes in mathematical reasoning
    # - "code_specialist": Cluster specializes in programming
    # - "creative_specialist": Cluster specializes in creative content
    # - "analytical_specialist": Cluster specializes in analysis
    # - "factual_specialist": Cluster specializes in facts/definitions
    # - "conversational_specialist": Cluster specializes in dialogue
    # - "generalist": Mixed specializations, jack-of-all-trades
    # - "mixed": Intentionally diverse (multiple specializations)
    
    # ========================================
    # MEMBER NODE TYPE DISTRIBUTION
    # ========================================
    
    node_type_distribution: {
        "math_specialist": 35,      # 35 nodes specialized in math
        "analytical_specialist": 8,  # 8 nodes specialized in analysis
        "generalist": 2              # 2 generalist nodes
        # Total: 45 nodes
    },
    
    # Diversity Metrics
    diversity: {
        type_diversity_score: 0.23,  # 0.0 = monoculture, 1.0 = maximum diversity
        specialization_focus: 0.78,  # How focused on primary type (0.0-1.0)
        balance_score: 0.85          # How well-balanced the distribution is
    },
    
    # ========================================
    # COLLECTIVE CHARACTERISTICS
    # ========================================
    
    collective_traits: {
        # Average chromosome traits across all member nodes
        average_ensemble_weight: 1.15,  # Cluster members slightly above average
        average_temperature: 0.25,       # Low temp = precise (math cluster)
        
        # Performance characteristics
        average_latency_seconds: 2.1,
        average_correctness: 0.89,
        average_confidence: 0.85,
        
        # Specialization strength
        primary_domain_accuracy: 0.92,   # Math tasks
        secondary_domain_accuracy: 0.84, # Logic/analysis tasks
        general_accuracy: 0.78           # Other tasks
    },
    
    # ========================================
    # EVOLUTIONARY PARAMETERS
    # ========================================
    
    evolution_config: {
        # Selection pressure (how aggressively to cull)
        selection_pressure: 0.20,  # Cull bottom 20% each generation
        
        # Mutation rates
        mutation_rate_per_node: 0.15,      # 15% of nodes mutate each gen
        crossover_rate: 0.10,              # 10% of spawned nodes are crossovers
        
        # Spawning strategy
        spawning_strategy: "replace_culled",  # Spawn to replace culled nodes
        spawning_bias: "fitness_weighted",     # Bias toward high-fitness parents
        
        # Diversity maintenance
        enforce_diversity: true,
        min_type_diversity: 0.15,  # At least 15% diversity required
        diversity_bonus: 0.05      # Fitness bonus for rare types
    },
    
    # ========================================
    # CLUSTER-LEVEL SPECIALIZATION
    # ========================================
    
    cluster_specialization: {
        # What this cluster is optimized for
        optimal_task_types: ["calculation", "proof", "word_problem", "equation_solving"],
        
        # Performance matrix (cluster's track record)
        task_type_performance: {
            "calculation": {
                "tasks_completed": 2847,
                "average_correctness": 0.94,
                "average_latency": 1.8
            },
            "proof": {
                "tasks_completed": 423,
                "average_correctness": 0.87,
                "average_latency": 3.2
            },
            "word_problem": {
                "tasks_completed": 1521,
                "average_correctness": 0.89,
                "average_latency": 2.5
            }
        },
        
        # Emergent behaviors (learned over time)
        emergent_strategies: [
            "parallel_verification_for_proofs",
            "cross_check_calculations_with_two_nodes",
            "route_complex_word_problems_to_analytical_specialists"
        ]
    },
    
    # ========================================
    # LINEAGE (Evolutionary History)
    # ========================================
    
    lineage: {
        generation: 7,
        parent_cluster: "cluster-math-parent-456" | null,
        
        # Major evolutionary events
        evolution_history: [
            {
                generation: 3,
                event_type: "type_shift",
                old_type: "generalist",
                new_type: "math_specialist",
                reason: "super_cluster_observed_high_fitness_on_math_tasks",
                nodes_affected: 28
            },
            {
                generation: 5,
                event_type: "population_expansion",
                old_size: 30,
                new_size: 45,
                reason: "high_demand_for_math_tasks"
            },
            {
                generation: 6,
                event_type: "diversity_injection",
                nodes_added: 8,
                types_added: ["analytical_specialist"],
                reason: "improve_word_problem_performance"
            }
        ],
        
        # Mutation tracking
        total_mutations_applied: 124,
        successful_mutations: 98,  # Mutations that improved fitness
        failed_mutations: 26,      # Mutations that hurt fitness (rolled back)
        
        # Spawning/Culling stats
        total_nodes_spawned: 67,
        total_nodes_culled: 37,
        current_generation_size: 45
    }
}

def compute_cluster_genome_hash(genome: ClusterGenome) -> str:
    """
    Compute genetic hash from cluster's collective genetic state
    Changes when: node type distribution changes, evolution config changes
    """
    genetic_data = {
        "cluster_type": genome.cluster_type,
        "node_type_distribution": genome.node_type_distribution,
        "generation": genome.lineage.generation,
        "average_ensemble_weight": round(genome.collective_traits.average_ensemble_weight, 2)
    }
    return f"clstr-{genome.cluster_type[:4]}-{hash(json.dumps(genetic_data, sort_keys=True))[:12]}"
```

### Runtime State

```python
ClusterRuntimeState = {
    # Execution Status
    status: "active" | "evolving" | "splitting" | "merging" | "dissolved",
    
    # Task Queue
    task_queue: {
        pending_tasks: 12,          # Tasks waiting to be assigned to nodes
        executing_tasks: 34,        # Tasks currently being executed by nodes
        completed_tasks: 2847,      # Total tasks completed (lifetime)
        failed_tasks: 89            # Tasks that failed
    },
    
    # Fitness Score - CRITICAL METRIC
    fitness: {
        current_score: 0.85,  # Current cluster fitness (0.0 to 1.0)
        
        # Fitness formula: (avg_correctness × avg_speed) / (avg_latency + compute_cost)
        components: {
            average_correctness: 0.89,      # Across all member nodes
            average_speed: 0.95,            # Tasks per hour normalized
            average_latency: 2.1,           # Seconds
            compute_cost_normalized: 0.18   # Normalized 0-1
        },
        
        score_history: [
            {"timestamp": t1, "score": 0.82},
            {"timestamp": t2, "score": 0.84},
            {"timestamp": t3, "score": 0.85}
        ],
        
        tasks_evaluated: 2847,  # Total tasks used to compute fitness
        last_updated: timestamp,
        
        # Fitness ranking
        percentile_in_super_cluster: 78,  # Top 22% of all clusters
        rank_in_super_cluster: 12          # Out of 55 clusters
    },
    
    # Node Health Monitoring
    node_health: {
        healthy_nodes: 42,          # Nodes with fitness > threshold
        struggling_nodes: 3,        # Nodes with declining fitness
        offline_nodes: 0,           # Nodes that disconnected
        
        # Node fitness distribution
        fitness_distribution: {
            "high_fitness (>0.85)": 15,
            "medium_fitness (0.70-0.85)": 27,
            "low_fitness (<0.70)": 3
        },
        
        # Nodes flagged for culling
        nodes_flagged_for_culling: ["node-abc", "node-def", "node-ghi"]
    },
    
    # Performance Metrics
    performance: {
        # Task execution stats
        throughput_tasks_per_hour: 234,
        average_task_latency: 2.1,
        success_rate: 0.97,
        
        # Verification stats
        verification_pass_rate: 0.94,
        verification_failures: 171,  # Tasks that failed verification
        
        # Node utilization
        average_node_utilization: 0.73,  # How busy nodes are on average
        idle_nodes: 8,
        busy_nodes: 37
    },
    
    # Resource Monitoring
    resources: {
        total_compute_capacity: 450,      # Normalized compute units
        utilized_compute: 328,            # Currently in use
        available_compute: 122,           # Available for new tasks
        compute_efficiency: 0.73          # Utilization ratio
    },
    
    # Network Connectivity
    network: {
        connected_to_super_cluster: true,
        connection_quality: "excellent" | "good" | "poor",
        average_latency_to_super_cluster_ms: 35,
        bandwidth_available_mbps: 1000
    },
    
    # Evolutionary State
    evolution: {
        last_evolution_cycle: timestamp,
        next_evolution_cycle: timestamp,
        evolution_frequency_hours: 24,     # Evolve every 24 hours
        mutations_since_last_cycle: 7,
        nodes_spawned_this_generation: 3,
        nodes_culled_this_generation: 3
    },
    
    # Active Cluster Genome
    active_genome: "clstr-math-a7f9e2d5c8b3"  # Reference to current genome hash
}
```

---

## Input Interfaces

### 1. Task Batch Assignment (from Super Cluster)

**Primary Input:** Receive a batch of simple queries to distribute among member nodes

```python
TaskBatchAssignment = {
    # Batch Identification
    batch_id: "batch-abc123",
    parent_query_id: "query-xyz789",
    assigned_by: "super-cluster-main",
    priority: "low" | "medium" | "high" | "critical",
    
    # The List of Simple Queries
    tasks: [
        {
            task_id: "task-001",
            query: "Calculate 15% of 240",
            classification: {
                task_type: "calculation",
                domain: "mathematics",
                complexity: "simple"
            },
            constraints: {
                max_latency_seconds: 3.0,
                min_confidence: 0.85
            }
        },
        {
            task_id: "task-002",
            query: "What is the area of a circle with radius 5?",
            classification: {
                task_type: "calculation",
                domain: "mathematics",
                complexity: "simple"
            },
            constraints: {
                max_latency_seconds: 3.0,
                min_confidence: 0.85
            }
        },
        {
            task_id: "task-003",
            query: "Solve for x: 2x + 5 = 13",
            classification: {
                task_type: "equation_solving",
                domain: "mathematics",
                complexity: "simple"
            },
            constraints: {
                max_latency_seconds: 5.0,
                min_confidence: 0.80
            }
        }
        # ... more tasks
    ],
    
    # Batch-level Requirements
    batch_requirements: {
        total_tasks: 15,
        expected_completion_time: 60,  # seconds
        verification_level: "standard" | "high" | "critical",
        redundancy_factor: 3  # How many nodes should answer each task
    },
    
    # Metadata
    timestamp: 1640995200,
    timeout_seconds: 120.0
}
```

**Input Channel:** gRPC stream from Super Cluster

**Acceptance Logic:**
```python
def should_accept_batch(batch: TaskBatchAssignment) -> bool:
    """
    Decide if cluster can handle this batch
    """
    # Check cluster status
    if self.runtime_state.status != "active":
        return False
    
    # Check if we have enough healthy nodes
    available_nodes = self.runtime_state.node_health.healthy_nodes
    required_nodes = batch.batch_requirements.total_tasks * batch.batch_requirements.redundancy_factor
    
    if available_nodes < required_nodes * 0.7:  # Need at least 70% capacity
        return False
    
    # Check if tasks match our specialization
    task_types = [task.classification.task_type for task in batch.tasks]
    our_optimal_types = self.genome.cluster_specialization.optimal_task_types
    
    match_ratio = sum(1 for t in task_types if t in our_optimal_types) / len(task_types)
    
    if match_ratio < 0.5:  # At least 50% should match our specialization
        return False
    
    return True
```

### 2. Evolutionary Signals (from Super Cluster)

**Input:** Receive fitness-based signals that trigger evolutionary changes

```python
EvolutionarySignal = {
    signal_id: "evo-signal-789",
    cluster_id: "cluster-math-specialists-001",
    signal_type: "fitness_feedback" | "population_adjustment" | "diversity_directive" | "type_shift_suggestion",
    
    # Fitness Feedback
    fitness_feedback: {
        current_fitness: 0.85,
        fitness_trend: "improving" | "stable" | "declining",
        percentile_rank: 78,  # Top 22%
        
        comparison: {
            super_cluster_average: 0.72,
            top_cluster_fitness: 0.93,
            delta_from_average: +0.13
        },
        
        signal: "positive" | "neutral" | "negative",
        
        recommendation: "continue" | "intensify_evolution" | "diversify" | "specialize_further"
    } | null,
    
    # Population Adjustment Directive
    population_adjustment: {
        current_size: 45,
        recommended_size: 55,
        reason: "high_demand_for_specialization",
        action: "expand" | "contract" | "maintain",
        
        expansion_plan: {
            nodes_to_spawn: 10,
            spawn_from_top_performers: true,
            mutation_strategy: "conservative"
        } | null,
        
        contraction_plan: {
            nodes_to_cull: 10,
            cull_bottom_performers: true
        } | null
    } | null,
    
    # Diversity Directive
    diversity_directive: {
        current_diversity: 0.23,
        recommended_diversity: 0.35,
        reason: "improve_word_problem_performance",
        
        action: "inject_diversity",
        
        diversity_injection_plan: {
            types_to_add: ["analytical_specialist"],
            quantity: 8,
            spawn_method: "crossover_with_external_cluster"
        }
    } | null,
    
    # Type Shift Suggestion
    type_shift_suggestion: {
        current_cluster_type: "generalist",
        suggested_cluster_type: "math_specialist",
        confidence: 0.87,
        reason: "cluster_shows_90%_fitness_on_math_tasks",
        
        transition_plan: {
            nodes_to_shift: 28,  # Shift 28 nodes from generalist to math specialist
            shift_gradually: true,
            generations_to_complete: 3
        }
    } | null,
    
    # Metadata
    timestamp: 1640995200,
    urgency: "low" | "medium" | "high"
}
```

**Input Channel:** gRPC RPC from Super Cluster

**Response Logic:**
```python
def handle_evolutionary_signal(signal: EvolutionarySignal):
    """
    Process evolutionary signal from Super Cluster
    """
    # Fitness Feedback
    if signal.fitness_feedback:
        self.update_fitness_state(signal.fitness_feedback)
        
        if signal.fitness_feedback.signal == "negative":
            # Trigger emergency evolution
            self.trigger_evolution_cycle(urgency="high")
    
    # Population Adjustment
    if signal.population_adjustment:
        if signal.population_adjustment.action == "expand":
            self.spawn_new_nodes(signal.population_adjustment.expansion_plan)
        elif signal.population_adjustment.action == "contract":
            self.cull_nodes(signal.population_adjustment.contraction_plan)
    
    # Diversity Directive
    if signal.diversity_directive:
        self.inject_diversity(signal.diversity_directive.diversity_injection_plan)
    
    # Type Shift Suggestion
    if signal.type_shift_suggestion:
        self.initiate_type_shift(signal.type_shift_suggestion.transition_plan)
```

### 3. Node Status Updates (from Member Nodes)

**Input:** Receive status reports from member nodes

```python
NodeStatusUpdate = {
    # Identification
    node_id: "node-550e8400",
    cluster_id: "cluster-math-specialists-001",
    update_type: "periodic" | "critical" | "availability_change",
    
    # Node Fitness
    fitness: {
        current_score: 0.87,
        tasks_evaluated: 147,
        recent_correctness: 0.91,
        recent_latency: 2.1
    },
    
    # Node Status
    status: "idle" | "executing" | "mutating" | "offline",
    current_task: "task-abc123" | null,
    
    # Resource Availability
    resources: {
        cpu_usage: 45.2,
        available_capacity: 0.55,  # 0.0 = fully busy, 1.0 = completely idle
        thermal_state: "normal" | "warm" | "hot",
        battery_level: 85 | null
    },
    
    # Performance Stats (since last update)
    performance: {
        tasks_completed: 12,
        tasks_failed: 1,
        average_latency: 2.3,
        average_correctness: 0.89
    },
    
    # Timestamp
    timestamp: 1640995200
}
```

**Input Channel:** gRPC stream from each member node (periodic updates every 30-60 seconds)

**Tracking Logic:**
```python
def handle_node_status_update(update: NodeStatusUpdate):
    """
    Track node health and availability
    """
    # Update node registry
    self.node_registry[update.node_id].update({
        "fitness": update.fitness.current_score,
        "status": update.status,
        "available_capacity": update.resources.available_capacity,
        "last_seen": update.timestamp
    })
    
    # Check if node health changed
    if update.fitness.current_score < 0.70:
        # Flag for potential culling
        self.flag_node_for_culling(update.node_id)
    
    # Update cluster-level metrics
    self.recalculate_cluster_fitness()
    self.update_node_health_stats()
```

### 4. Verification Results (from Super Cluster)

**Input:** Receive verification results for tasks completed by cluster

```python
VerificationResult = {
    # Identification
    verification_id: "verify-abc123",
    task_id: "task-001",
    batch_id: "batch-abc123",
    cluster_id: "cluster-math-specialists-001",
    
    # Node Answers (from redundancy)
    node_answers: [
        {
            node_id: "node-550e8400",
            answer: "36",
            confidence: 0.92,
            latency: 1.8,
            ensemble_weight: 1.2
        },
        {
            node_id: "node-661f9511",
            answer: "36",
            confidence: 0.88,
            latency: 2.1,
            ensemble_weight: 1.1
        },
        {
            node_id: "node-772g0622",
            answer: "36",
            confidence: 0.85,
            latency: 2.3,
            ensemble_weight: 1.0
        }
    ],
    
    # Verification Outcome
    verification: {
        method: "programmatic" | "redundancy_voting" | "super_llm_check",
        
        # Correctness determination
        correctness: 1.0,  # 0.0 = wrong, 1.0 = perfect
        ground_truth: "36" | null,  # If available
        
        # Consensus analysis
        consensus: "unanimous" | "majority" | "split",
        agreement_percentage: 100,  # All 3 nodes agreed
        
        # Verification details
        verified_by: "super-cluster-main",
        verified_at: timestamp,
        verification_latency: 0.5  # seconds to verify
    },
    
    # Fitness Impact (for each node)
    fitness_updates: [
        {
            node_id: "node-550e8400",
            old_fitness: 0.85,
            new_fitness: 0.87,
            delta: +0.02,
            reason: "correct_answer_fast_response"
        },
        {
            node_id: "node-661f9511",
            old_fitness: 0.82,
            new_fitness: 0.83,
            delta: +0.01,
            reason: "correct_answer_average_latency"
        },
        {
            node_id: "node-772g0622",
            old_fitness: 0.80,
            new_fitness: 0.81,
            delta: +0.01,
            reason: "correct_answer_slow_response"
        }
    ],
    
    # Cluster-level Feedback
    cluster_feedback: {
        task_type_performance_update: {
            task_type: "calculation",
            new_average_correctness: 0.94,
            new_average_latency: 2.0
        },
        
        cluster_fitness_impact: {
            old_cluster_fitness: 0.84,
            new_cluster_fitness: 0.85,
            delta: +0.01
        }
    }
}
```

**Input Channel:** gRPC RPC from Super Cluster (after verification completes)

**Response Logic:**
```python
def handle_verification_result(result: VerificationResult):
    """
    Process verification result and update node/cluster fitness
    """
    # Update individual node fitness scores
    for fitness_update in result.fitness_updates:
        self.update_node_fitness(
            node_id=fitness_update.node_id,
            new_fitness=fitness_update.new_fitness
        )
    
    # Update cluster performance matrix
    self.update_task_type_performance(
        task_type=result.cluster_feedback.task_type_performance_update.task_type,
        correctness=result.cluster_feedback.task_type_performance_update.new_average_correctness,
        latency=result.cluster_feedback.task_type_performance_update.new_average_latency
    )
    
    # Update cluster fitness
    self.runtime_state.fitness.current_score = result.cluster_feedback.cluster_fitness_impact.new_cluster_fitness
    
    # Track verification stats
    self.runtime_state.performance.verification_pass_rate = self.calculate_verification_pass_rate()
    
    # Check if any nodes should be flagged for culling
    for fitness_update in result.fitness_updates:
        if fitness_update.new_fitness < self.CULLING_THRESHOLD:
            self.flag_node_for_culling(fitness_update.node_id)
```

---

## Output Interfaces

### 1. Task Results (to Super Cluster)

**Primary Output:** Send aggregated answers after verification

```python
TaskBatchResult = {
    # Identification
    batch_id: "batch-abc123",
    cluster_id: "cluster-math-specialists-001",
    parent_query_id: "query-xyz789",
    
    # Aggregated Results for Each Task
    task_results: [
        {
            task_id: "task-001",
            
            # Node Answers (raw)
            node_answers: [
                {
                    node_id: "node-550e8400",
                    answer: "36",
                    confidence: 0.92,
                    ensemble_weight: 1.2,
                    latency: 1.8
                },
                {
                    node_id: "node-661f9511",
                    answer: "36",
                    confidence: 0.88,
                    ensemble_weight: 1.1,
                    latency: 2.1
                },
                {
                    node_id: "node-772g0622",
                    answer: "36",
                    confidence: 0.85,
                    ensemble_weight: 1.0,
                    latency: 2.3
                }
            ],
            
            # Cluster Verification (internal)
            cluster_verification: {
                method: "redundancy_voting",
                consensus: "unanimous",
                agreement: 100,
                verification_passed: true
            },
            
            # Aggregated Answer (cluster's consensus)
            aggregated_answer: {
                answer: "36",
                confidence: 0.88,  # Average confidence
                ensemble_weighted_confidence: 1.02,  # Weighted by ensemble weights
                latency: 2.1,  # Max latency among nodes
                consensus_strength: "unanimous",
                
                reasoning: "All 3 nodes produced identical answer '36' with high confidence"
            }
        },
        {
            task_id: "task-002",
            node_answers: [...],
            cluster_verification: {...},
            aggregated_answer: {
                answer: "78.54",
                confidence: 0.85,
                ensemble_weighted_confidence: 0.98,
                latency: 2.5,
                consensus_strength: "majority",
                reasoning: "2 of 3 nodes agreed on '78.54', 1 node gave '78.5' (close enough)"
            }
        },
        {
            task_id: "task-003",
            node_answers: [...],
            cluster_verification: {...},
            aggregated_answer: {
                answer: "x = 4",
                confidence: 0.91,
                ensemble_weighted_confidence: 1.05,
                latency: 2.8,
                consensus_strength: "unanimous",
                reasoning: "All 3 nodes produced identical answer 'x = 4'"
            }
        }
        // ... more task results
    ],
    
    # Batch-level Metrics
    batch_metrics: {
        total_tasks: 15,
        completed_tasks: 15,
        failed_tasks: 0,
        
        average_latency: 2.3,
        average_confidence: 0.87,
        average_consensus_strength: 0.92,
        
        verification_pass_rate: 1.0,  # All tasks passed internal verification
        
        total_execution_time: 45.2  # seconds
    },
    
    # Cluster Performance
    cluster_performance: {
        cluster_fitness: 0.85,
        cluster_type: "math_specialist",
        
        node_utilization: {
            nodes_used: 9,  # 9 nodes participated (3 per task, 3 tasks in parallel)
            average_node_load: 0.67
        }
    },
    
    # Send to Super Cluster for final verification
    send_to: "super-cluster-main",
    
    # Timestamp
    timestamp: 1640995247.2
}
```

**Output Channel:** gRPC response to Super Cluster

**Aggregation Logic:**
```python
def aggregate_task_results(task_id: str, node_answers: List[NodeAnswer]) -> AggregatedAnswer:
    """
    Combine answers from multiple nodes into cluster consensus
    Uses ensemble weighting for voting
    """
    # Exact match check
    unique_answers = set(answer.answer for answer in node_answers)
    
    if len(unique_answers) == 1:
        # Unanimous agreement
        return AggregatedAnswer(
            answer=node_answers[0].answer,
            confidence=sum(a.confidence for a in node_answers) / len(node_answers),
            ensemble_weighted_confidence=sum(a.confidence * a.ensemble_weight for a in node_answers) / sum(a.ensemble_weight for a in node_answers),
            latency=max(a.latency for a in node_answers),
            consensus_strength="unanimous",
            reasoning=f"All {len(node_answers)} nodes produced identical answer"
        )
    
    # Semantic similarity check for near-matches
    answer_groups = self.group_by_semantic_similarity(node_answers)
    
    # Find majority answer (weighted by ensemble weights)
    majority_group = max(answer_groups, key=lambda g: sum(a.ensemble_weight for a in g))
    
    if len(majority_group) >= len(node_answers) * 0.6:  # At least 60% agree
        return AggregatedAnswer(
            answer=majority_group[0].answer,
            confidence=sum(a.confidence for a in majority_group) / len(majority_group),
            ensemble_weighted_confidence=sum(a.confidence * a.ensemble_weight for a in majority_group) / sum(a.ensemble_weight for a in majority_group),
            latency=max(a.latency for a in node_answers),
            consensus_strength="majority",
            reasoning=f"{len(majority_group)} of {len(node_answers)} nodes agreed"
        )
    
    # Split vote - use highest ensemble-weighted confidence
    best_answer = max(node_answers, key=lambda a: a.confidence * a.ensemble_weight)
    
    return AggregatedAnswer(
        answer=best_answer.answer,
        confidence=best_answer.confidence,
        ensemble_weighted_confidence=best_answer.confidence * best_answer.ensemble_weight,
        latency=max(a.latency for a in node_answers),
        consensus_strength="split",
        reasoning=f"Split vote - selected answer with highest ensemble-weighted confidence ({best_answer.confidence * best_answer.ensemble_weight:.2f})"
    )
```

### 2. Performance Reports (to Super Cluster)

**Output:** Regular cluster performance and health updates

```python
ClusterPerformanceReport = {
    # Identification
    cluster_id: "cluster-math-specialists-001",
    report_type: "periodic" | "on_demand" | "critical",
    reporting_period: {
        start: 1640995200,
        end: 1640998800,
        duration_hours: 1.0
    },
    
    # Cluster Fitness
    fitness: {
        current_score: 0.85,
        trend: "improving" | "stable" | "declining",
        change_since_last_report: +0.03,
        
        components: {
            average_correctness: 0.89,
            average_latency: 2.1,
            average_speed: 0.95,
            compute_cost: 0.18
        }
    },
    
    # Task Execution Statistics
    tasks: {
        batches_received: 23,
        batches_completed: 23,
        batches_failed: 0,
        
        total_tasks: 347,
        completed_tasks: 342,
        failed_tasks: 5,
        success_rate: 0.986,
        
        # By Task Type
        by_type: {
            "calculation": {
                completed: 203,
                avg_correctness: 0.94,
                avg_latency: 1.8
            },
            "equation_solving": {
                completed: 89,
                avg_correctness: 0.87,
                avg_latency: 2.5
            },
            "word_problem": {
                completed: 50,
                avg_correctness: 0.82,
                avg_latency: 3.2
            }
        }
    },
    
    # Node Population Health
    node_health: {
        total_nodes: 45,
        healthy_nodes: 42,
        struggling_nodes: 3,
        offline_nodes: 0,
        
        fitness_distribution: {
            "high (>0.85)": 15,
            "medium (0.70-0.85)": 27,
            "low (<0.70)": 3
        },
        
        average_node_fitness: 0.81,
        top_node_fitness: 0.94,
        bottom_node_fitness: 0.67
    },
    
    # Evolutionary Activity
    evolution: {
        last_evolution_cycle: timestamp_24h_ago,
        mutations_applied: 7,
        nodes_spawned: 3,
        nodes_culled: 3,
        
        genetic_diversity: 0.23,
        specialization_focus: 0.78
    },
    
    # Resource Utilization
    resources: {
        total_compute_capacity: 450,
        utilized_compute: 328,
        compute_efficiency: 0.73,
        
        average_node_utilization: 0.73,
        peak_utilization: 0.92,
        idle_capacity: 122
    },
    
    # Recommendations
    recommendations: {
        population_adjustment: "expand_by_10" | "contract_by_5" | "maintain",
        diversity_needs: "inject_analytical_specialists" | "maintain_diversity",
        evolution_urgency: "low" | "medium" | "high",
        
        reasoning: "Cluster performing well on math tasks but struggling with word problems. Recommend adding 8 analytical specialists."
    }
}
```

**Output Channel:** gRPC RPC to Super Cluster (periodic, every 30-60 minutes)

### 3. Node Fitness Updates (to Member Nodes)

**Output:** Send fitness feedback to nodes after verification

```python
NodeFitnessUpdate = {
    # Identification
    update_id: "fitness-update-abc123",
    node_id: "node-550e8400",
    cluster_id: "cluster-math-specialists-001",
    
    # Task Context
    task_id: "task-001",
    task_type: "calculation",
    
    # Verification Outcome
    verification: {
        correctness: 1.0,  # Node was correct
        verified_by: "cluster_internal_verification",
        cluster_consensus: "unanimous",  # All nodes agreed
        latency_actual: 1.8
    },
    
    # Fitness Update Calculation
    fitness_calculation: {
        old_fitness_score: 0.85,
        new_fitness_score: 0.87,
        delta: +0.02,
        
        reason: "correct_answer_fast_response",
        
        # How fitness was calculated
        formula: "(correctness × speed) / (latency + cost)",
        components: {
            correctness: 1.0,
            speed: 1.1,  # Faster than average
            latency: 1.8,
            cost: 0.15
        }
    },
    
    # Cluster Context (for node's self-assessment)
    cluster_context: {
        cluster_average_fitness: 0.81,
        node_percentile: 72,  # This node is top 28%
        node_rank: 13,  # Out of 45 nodes
        
        signal: "positive" | "neutral" | "negative"
    },
    
    # Credits Earned
    credits_earned: {
        amount: 45,
        calculation: {
            base_rate: 30,
            quality_multiplier: 1.3,  # Based on correctness = 1.0
            latency_bonus: 1.1,       # Fast response
            difficulty_multiplier: 1.05
        }
    },
    
    # Timestamp
    timestamp: 1640995203.5
}
```

**Output Channel:** gRPC RPC to each node that participated in the task

### 4. Evolution Directives (to Member Nodes)

**Output:** Trigger mutations, spawning, or culling in nodes

```python
EvolutionDirective = {
    # Identification
    directive_id: "evo-directive-789",
    cluster_id: "cluster-math-specialists-001",
    directive_type: "mutation" | "spawn" | "cull" | "type_shift",
    
    # Target Nodes
    target_nodes: ["node-550e8400", "node-661f9511"],
    
    # Mutation Directive
    mutation: {
        mutation_type: "weight_adjustment" | "type_refinement" | "parameter_tweak",
        
        target_node: "node-550e8400",
        current_fitness: 0.94,
        reason: "top_performer_increase_weight",
        
        mutation_params: {
            ensemble_weight: 1.2 -> 1.5,  # Increase voting weight
            type_config_refinement: {
                add_few_shot_examples: [...successful_examples...]
            }
        }
    } | null,
    
    # Spawn Directive
    spawn: {
        parent_node: "node-550e8400",  # High-fitness parent
        spawn_method: "mutation" | "crossover",
        
        spawn_params: {
            node_type: "math_specialist",  # Same as parent
            ensemble_weight: 1.3,  # Inherit from parent
            mutation_applied: "weight_adjustment"
        },
        
        reason: "replace_culled_node"
    } | null,
    
    # Cull Directive
    cull: {
        target_node: "node-772g0622",
        current_fitness: 0.65,  # Below threshold
        reason: "low_fitness_bottom_20%",
        
        cull_method: "immediate" | "graceful",  # Graceful = finish current task first
        
        replacement_plan: {
            spawn_from_parent: "node-550e8400",
            mutation_type: "parameter_tweak"
        }
    } | null,
    
    # Type Shift Directive
    type_shift: {
        target_node: "node-661f9511",
        old_type: "generalist",
        new_type: "math_specialist",
        reason: "cluster_needs_more_math_specialists",
        
        transition_plan: {
            gradual: false,  # Immediate shift
            apply_archetype: NODE_TYPE_ARCHETYPES["math_specialist"]
        }
    } | null,
    
    # Metadata
    generation: 8,
    urgency: "low" | "medium" | "high",
    timestamp: 1640995300
}
```

**Output Channel:** gRPC RPC to each target node

---

## Internal Mechanisms

### 1. Task Distribution Pipeline

**Core Process: Receive batch, assign to nodes, verify answers, aggregate, send to Super Cluster**

```python
class TaskDistributor:
    def distribute_batch(self, batch: TaskBatchAssignment) -> TaskBatchResult:
        """
        Main task distribution: assign to nodes, verify, aggregate
        """
        start_time = time.now()
        
        # Step 1: Validate batch acceptance
        if not self.should_accept_batch(batch):
            return BatchRejection(reason="insufficient_capacity")
        
        # Step 2: Add batch to queue
        self.runtime_state.task_queue.pending_tasks += len(batch.tasks)
        
        # Step 3: For each task, assign to N nodes (redundancy)
        task_assignments = []
        redundancy_factor = batch.batch_requirements.redundancy_factor  # e.g., 3
        
        for task in batch.tasks:
            # Select N appropriate nodes for this task
            selected_nodes = self.select_nodes_for_task(
                task=task,
                count=redundancy_factor
            )
            
            # Send task to each selected node
            task_assignments.append({
                "task_id": task.task_id,
                "assigned_nodes": selected_nodes,
                "task": task
            })
            
            for node_id in selected_nodes:
                self.send_task_to_node(node_id, task)
        
        # Step 4: Wait for all node responses (with timeout)
        node_responses = await self.collect_node_responses(
            task_assignments=task_assignments,
            timeout=batch.timeout_seconds
        )
        
        # Step 5: Verify and aggregate each task's answers
        task_results = []
        
        for task_assignment in task_assignments:
            task_id = task_assignment["task_id"]
            
            # Get all node answers for this task
            node_answers = node_responses[task_id]
            
            # Internal verification (redundancy voting)
            verification = self.verify_answers(node_answers)
            
            # Aggregate answers into cluster consensus
            aggregated_answer = self.aggregate_task_results(task_id, node_answers)
            
            task_results.append({
                "task_id": task_id,
                "node_answers": node_answers,
                "cluster_verification": verification,
                "aggregated_answer": aggregated_answer
            })
        
        # Step 6: Package batch result
        batch_result = TaskBatchResult(
            batch_id=batch.batch_id,
            cluster_id=self.cluster_id,
            parent_query_id=batch.parent_query_id,
            task_results=task_results,
            batch_metrics=self.calculate_batch_metrics(task_results),
            cluster_performance=self.get_cluster_performance_snapshot(),
            send_to=batch.assigned_by,
            timestamp=time.now()
        )
        
        # Step 7: Send batch result to Super Cluster
        self.send_batch_result_to_super_cluster(batch_result)
        
        # Step 8: Update cluster state
        self.runtime_state.task_queue.pending_tasks -= len(batch.tasks)
        self.runtime_state.task_queue.completed_tasks += len(batch.tasks)
        
        return batch_result
    
    def select_nodes_for_task(self, task: Task, count: int) -> List[str]:
        """
        Select the best N nodes for this specific task
        Uses specialization matching and load balancing
        """
        # Get all healthy nodes
        available_nodes = [
            node_id for node_id, node in self.node_registry.items()
            if node["status"] == "idle" and node["fitness"] > 0.70
        ]
        
        # Score each node for this task
        node_scores = []
        for node_id in available_nodes:
            node = self.node_registry[node_id]
            
            # Base score = fitness
            score = node["fitness"]
            
            # Bonus for specialization match
            if self.is_specialization_match(node["node_type"], task.classification.task_type):
                score += 0.15
            
            # Penalty for high current load
            if node["available_capacity"] < 0.3:
                score -= 0.10
            
            # Bonus for high ensemble weight
            score += (node.get("ensemble_weight", 1.0) - 1.0) * 0.1
            
            node_scores.append((node_id, score))
        
        # Sort by score and select top N
        node_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [node_id for node_id, score in node_scores[:count]]
        
        # If we don't have enough nodes, fill with random selection
        if len(selected) < count:
            remaining = [n for n in available_nodes if n not in selected]
            selected.extend(random.sample(remaining, min(count - len(selected), len(remaining))))
        
        return selected
    
    async def collect_node_responses(self, task_assignments: List, timeout: float) -> Dict:
        """
        Collect responses from all assigned nodes
        """
        responses = {}
        
        # Create futures for all node responses
        futures = {}
        for assignment in task_assignments:
            task_id = assignment["task_id"]
            responses[task_id] = []
            
            for node_id in assignment["assigned_nodes"]:
                future = self.wait_for_node_response(node_id, task_id, timeout)
                futures[(task_id, node_id)] = future
        
        # Wait for all futures
        for (task_id, node_id), future in futures.items():
            try:
                node_response = await asyncio.wait_for(future, timeout=timeout)
                responses[task_id].append(node_response)
            except asyncio.TimeoutError:
                # Node didn't respond in time
                self.flag_node_timeout(node_id)
        
        return responses
    
    def verify_answers(self, node_answers: List[NodeAnswer]) -> VerificationResult:
        """
        Internal cluster verification using redundancy voting
        """
        # Check for unanimous agreement
        unique_answers = set(answer.answer for answer in node_answers)
        
        if len(unique_answers) == 1:
            return VerificationResult(
                method="redundancy_voting",
                consensus="unanimous",
                agreement=100,
                verification_passed=True
            )
        
        # Check for majority agreement (weighted by ensemble weights)
        answer_votes = {}
        for answer in node_answers:
            answer_text = answer.answer
            weight = answer.ensemble_weight
            answer_votes[answer_text] = answer_votes.get(answer_text, 0) + weight
        
        # Find majority
        majority_answer = max(answer_votes, key=answer_votes.get)
        majority_weight = answer_votes[majority_answer]
        total_weight = sum(answer_votes.values())
        agreement_pct = (majority_weight / total_weight) * 100
        
        if agreement_pct >= 60:
            return VerificationResult(
                method="redundancy_voting",
                consensus="majority",
                agreement=agreement_pct,
                verification_passed=True
            )
        
        # Split vote
        return VerificationResult(
            method="redundancy_voting",
            consensus="split",
            agreement=agreement_pct,
            verification_passed=False  # Escalate to Super Cluster for verification
        )
```

### 2. Fitness Management

```python
class FitnessManager:
    def calculate_cluster_fitness(self) -> float:
        """
        Calculate cluster fitness from member nodes
        Formula: (avg_correctness × avg_speed) / (avg_latency + compute_cost)
        """
        # Get all healthy nodes
        healthy_nodes = [
            node for node in self.node_registry.values()
            if node["status"] != "offline"
        ]
        
        if not healthy_nodes:
            return 0.0
        
        # Calculate averages
        avg_correctness = sum(node["recent_correctness"] for node in healthy_nodes) / len(healthy_nodes)
        avg_latency = sum(node["recent_latency"] for node in healthy_nodes) / len(healthy_nodes)
        
        # Speed = tasks per hour (normalized)
        total_tasks = self.runtime_state.task_queue.completed_tasks
        hours_active = (time.now() - self.created_at) / 3600
        avg_speed = total_tasks / hours_active if hours_active > 0 else 0
        avg_speed_normalized = min(avg_speed / 100, 1.0)  # Normalize to 0-1
        
        # Compute cost (normalized)
        total_compute = sum(node.get("compute_used", 0) for node in healthy_nodes)
        avg_compute_cost = (total_compute / len(healthy_nodes)) / 100  # Normalize
        
        # Fitness formula
        fitness = (avg_correctness * avg_speed_normalized) / (1 + avg_latency + avg_compute_cost)
        
        return min(fitness, 1.0)  # Cap at 1.0
    
    def update_node_fitness(self, node_id: str, new_fitness: float):
        """
        Update individual node fitness and recalculate cluster fitness
        """
        # Update node registry
        self.node_registry[node_id]["fitness"] = new_fitness
        
        # Recalculate cluster fitness
        self.runtime_state.fitness.current_score = self.calculate_cluster_fitness()
        self.runtime_state.fitness.last_updated = time.now()
        
        # Add to history
        self.runtime_state.fitness.score_history.append({
            "timestamp": time.now(),
            "score": self.runtime_state.fitness.current_score
        })
        
        # Check if node should be flagged for culling
        if new_fitness < self.CULLING_THRESHOLD:
            self.flag_node_for_culling(node_id)
    
    def flag_node_for_culling(self, node_id: str):
        """
        Flag a node for potential culling in next evolution cycle
        """
        if node_id not in self.runtime_state.node_health.nodes_flagged_for_culling:
            self.runtime_state.node_health.nodes_flagged_for_culling.append(node_id)
            
            # Update struggling nodes count
            self.runtime_state.node_health.struggling_nodes += 1
```

### 3. Evolutionary Engine

```python
class EvolutionaryEngine:
    def trigger_evolution_cycle(self, urgency: str = "normal"):
        """
        Main evolutionary cycle: selection, culling, spawning, mutation
        """
        print(f"[Cluster {self.cluster_id}] Starting evolution cycle (generation {self.genome.generation + 1})")
        
        # Step 1: Evaluate all nodes
        node_fitness_scores = self.evaluate_all_nodes()
        
        # Step 2: Selection - identify top and bottom performers
        sorted_nodes = sorted(node_fitness_scores.items(), key=lambda x: x[1], reverse=True)
        
        population_size = len(sorted_nodes)
        cull_count = int(population_size * self.genome.evolution_config.selection_pressure)  # 20%
        
        top_performers = sorted_nodes[:cull_count]  # Top 20%
        bottom_performers = sorted_nodes[-cull_count:]  # Bottom 20%
        
        # Step 3: Culling - remove bottom performers
        for node_id, fitness in bottom_performers:
            self.cull_node(node_id, reason="low_fitness_bottom_20%")
        
        # Step 4: Spawning - replace culled nodes
        for i in range(cull_count):
            # Select parent from top performers (fitness-weighted selection)
            parent_node_id = self.select_parent(top_performers)
            
            # Decide mutation type
            mutation_type = self.choose_mutation_type()
            
            # Spawn new node
            self.spawn_node(parent_node_id, mutation_type)
        
        # Step 5: Mutation - apply mutations to some surviving nodes
        mutation_count = int(population_size * self.genome.evolution_config.mutation_rate_per_node)
        
        for i in range(mutation_count):
            # Select random survivor (not top or bottom)
            middle_performers = sorted_nodes[cull_count:-cull_count]
            if middle_performers:
                node_id, _ = random.choice(middle_performers)
                mutation_type = self.choose_mutation_type()
                self.mutate_node(node_id, mutation_type)
        
        # Step 6: Diversity check
        if self.genome.evolution_config.enforce_diversity:
            self.ensure_diversity()
        
        # Step 7: Update cluster genome
        self.update_cluster_genome()
        
        # Step 8: Increment generation
        self.genome.lineage.generation += 1
        self.runtime_state.evolution.last_evolution_cycle = time.now()
        
        print(f"[Cluster {self.cluster_id}] Evolution cycle complete. New generation: {self.genome.lineage.generation}")
    
    def cull_node(self, node_id: str, reason: str):
        """
        Remove a node from the cluster (send cull directive)
        """
        # Send cull directive to node
        directive = EvolutionDirective(
            directive_type="cull",
            target_nodes=[node_id],
            cull={
                "target_node": node_id,
                "current_fitness": self.node_registry[node_id]["fitness"],
                "reason": reason,
                "cull_method": "graceful"
            }
        )
        
        self.send_evolution_directive(directive)
        
        # Update cluster state
        self.genome.member_nodes.remove(node_id)
        del self.node_registry[node_id]
        self.genome.population.current_size -= 1
        
        # Track in lineage
        self.genome.lineage.total_nodes_culled += 1
        
        print(f"[Cluster {self.cluster_id}] Culled node {node_id}: {reason}")
    
    def spawn_node(self, parent_node_id: str, mutation_type: str):
        """
        Spawn a new node from a parent (send spawn directive)
        """
        parent_node = self.node_registry[parent_node_id]
        
        # Send spawn directive to parent node's owner
        directive = EvolutionDirective(
            directive_type="spawn",
            target_nodes=[parent_node_id],
            spawn={
                "parent_node": parent_node_id,
                "spawn_method": "mutation",
                "spawn_params": {
                    "node_type": parent_node["node_type"],
                    "ensemble_weight": parent_node.get("ensemble_weight", 1.0),
                    "mutation_applied": mutation_type
                },
                "reason": "replace_culled_node"
            }
        )
        
        self.send_evolution_directive(directive)
        
        # The new node will register itself when it starts up
        # Update cluster state (will be finalized when node registers)
        self.genome.lineage.total_nodes_spawned += 1
        
        print(f"[Cluster {self.cluster_id}] Spawned new node from parent {parent_node_id} with mutation {mutation_type}")
    
    def mutate_node(self, node_id: str, mutation_type: str):
        """
        Apply mutation to an existing node (send mutation directive)
        """
        node = self.node_registry[node_id]
        
        # Determine mutation parameters based on type
        mutation_params = self.generate_mutation_params(node, mutation_type)
        
        # Send mutation directive
        directive = EvolutionDirective(
            directive_type="mutation",
            target_nodes=[node_id],
            mutation={
                "mutation_type": mutation_type,
                "target_node": node_id,
                "current_fitness": node["fitness"],
                "reason": "evolution_cycle_improvement",
                "mutation_params": mutation_params
            }
        )
        
        self.send_evolution_directive(directive)
        
        # Track in lineage
        self.genome.lineage.total_mutations_applied += 1
        
        print(f"[Cluster {self.cluster_id}] Mutated node {node_id} with {mutation_type}")
    
    def choose_mutation_type(self) -> str:
        """
        Randomly select mutation type based on distribution
        """
        from DELLM.md: 
        MUTATION_DISTRIBUTION = {
            "type_shift": 0.50,          # 50%
            "weight_adjustment": 0.30,    # 30%
            "type_refinement": 0.15,      # 15%
            "parameter_tweak": 0.05       # 5%
        }
        
        rand = random.random()
        if rand < 0.50:
            return "type_shift"
        elif rand < 0.80:
            return "weight_adjustment"
        elif rand < 0.95:
            return "type_refinement"
        else:
            return "parameter_tweak"
    
    def ensure_diversity(self):
        """
        Ensure cluster maintains minimum diversity
        If too homogeneous, inject diverse node types
        """
        current_diversity = self.calculate_diversity()
        min_diversity = self.genome.evolution_config.min_type_diversity
        
        if current_diversity < min_diversity:
            # Need to inject diversity
            inject_count = max(3, int(self.genome.population.current_size * 0.1))
            
            # Determine which types are underrepresented
            all_types = ["math_specialist", "code_specialist", "analytical_specialist", 
                        "creative_specialist", "factual_specialist", "conversational_specialist"]
            
            current_types = self.genome.node_type_distribution.keys()
            missing_types = [t for t in all_types if t not in current_types]
            
            # Randomly select some low-fitness nodes to shift type
            low_fitness_nodes = [
                node_id for node_id, node in self.node_registry.items()
                if node["fitness"] < 0.75
            ]
            
            nodes_to_shift = random.sample(low_fitness_nodes, min(inject_count, len(low_fitness_nodes)))
            
            for node_id in nodes_to_shift:
                new_type = random.choice(missing_types)
                
                # Send type shift directive
                directive = EvolutionDirective(
                    directive_type="type_shift",
                    target_nodes=[node_id],
                    type_shift={
                        "target_node": node_id,
                        "old_type": self.node_registry[node_id]["node_type"],
                        "new_type": new_type,
                        "reason": "diversity_injection"
                    }
                )
                
                self.send_evolution_directive(directive)
            
            print(f"[Cluster {self.cluster_id}] Injected diversity: shifted {len(nodes_to_shift)} nodes to {missing_types}")
    
    def calculate_diversity(self) -> float:
        """
        Calculate cluster's type diversity score
        0.0 = monoculture (all same type)
        1.0 = maximum diversity (equal distribution)
        """
        type_counts = list(self.genome.node_type_distribution.values())
        total_nodes = sum(type_counts)
        
        if total_nodes == 0:
            return 0.0
        
        # Shannon entropy
        diversity = 0.0
        for count in type_counts:
            if count > 0:
                p = count / total_nodes
                diversity -= p * math.log2(p)
        
        # Normalize to 0-1
        max_diversity = math.log2(len(type_counts)) if len(type_counts) > 1 else 1.0
        return diversity / max_diversity if max_diversity > 0 else 0.0
```

### 4. Cluster Lifecycle Management

```python
class ClusterLifecycle:
    def check_cluster_health(self):
        """
        Monitor cluster health and trigger splits/merges/dissolution
        """
        # Check population size
        current_size = self.genome.population.current_size
        
        # Too small - consider merging or dissolving
        if current_size < self.genome.population.min_size:
            self.trigger_cluster_dissolution()
        
        # Too large - consider splitting
        elif current_size > self.genome.population.max_size:
            self.trigger_cluster_split()
        
        # Check fitness
        if self.runtime_state.fitness.current_score < 0.60:
            # Cluster struggling - increase evolution urgency
            self.trigger_evolution_cycle(urgency="high")
    
    def trigger_cluster_split(self):
        """
        Split cluster into two smaller clusters
        """
        print(f"[Cluster {self.cluster_id}] Splitting cluster (size: {self.genome.population.current_size})")
        
        # Divide nodes by specialization
        type_groups = {}
        for node_id, node in self.node_registry.items():
            node_type = node["node_type"]
            if node_type not in type_groups:
                type_groups[node_type] = []
            type_groups[node_type].append(node_id)
        
        # Create two new clusters
        # Cluster A: Primary specialization
        cluster_a_nodes = []
        # Cluster B: Secondary specializations
        cluster_b_nodes = []
        
        # Split logic (example: by type dominance)
        sorted_types = sorted(type_groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Cluster A gets most common type
        cluster_a_nodes.extend(sorted_types[0][1])
        
        # Cluster B gets remaining types
        for node_type, nodes in sorted_types[1:]:
            cluster_b_nodes.extend(nodes)
        
        # Create new cluster identities
        cluster_a_id = f"cluster-{sorted_types[0][0]}-{generate_uuid()[:8]}"
        cluster_b_id = f"cluster-mixed-{generate_uuid()[:8]}"
        
        # Notify Super Cluster of split
        self.notify_super_cluster_split(
            cluster_a_id=cluster_a_id,
            cluster_a_nodes=cluster_a_nodes,
            cluster_b_id=cluster_b_id,
            cluster_b_nodes=cluster_b_nodes
        )
        
        # Dissolve this cluster
        self.runtime_state.status = "dissolved"
    
    def trigger_cluster_dissolution(self):
        """
        Dissolve cluster if too small
        Nodes are reassigned to other clusters
        """
        print(f"[Cluster {self.cluster_id}] Dissolving cluster (size: {self.genome.population.current_size} < min: {self.genome.population.min_size})")
        
        # Notify Super Cluster
        self.notify_super_cluster_dissolution(
            cluster_id=self.cluster_id,
            member_nodes=list(self.genome.member_nodes)
        )
        
        # Super Cluster will reassign nodes to other clusters
        self.runtime_state.status = "dissolved"
```

---

## Summary

The Cluster Block is the coordination layer between Super Cluster and Nodes:

**Core Components:**
- **Cluster Identity** - Unique ID, membership, specialization type
- **Cluster Genome** - Collective genetic profile of all member nodes
- **Runtime State** - Fitness, task queue, node health monitoring
- **Evolutionary Engine** - Selection, mutation, spawning, culling logic

**Key Inputs:**
1. Task batch assignments from Super Cluster
2. Evolutionary signals from Super Cluster
3. Node status updates from member nodes
4. Verification results from Super Cluster

**Key Outputs:**
1. Aggregated task results to Super Cluster
2. Performance reports and cluster health metrics
3. Fitness updates to member nodes
4. Evolution directives (mutation, spawn, cull)

**Internal Mechanisms:**
- Task distribution with redundancy voting
- Internal verification using ensemble weighting
- Fitness calculation and node ranking
- Evolutionary cycles (selection, culling, spawning, mutation)
- Diversity maintenance
- Cluster lifecycle (split, merge, dissolve)

**Key Principles:**
- **No LLM** - Pure coordination logic
- **Symbiotic group** - Nodes work together, share specializations
- **Evolutionary pressure** - Bottom 20% culled each generation
- **Fitness-driven** - All decisions based on performance metrics
- **Ensemble intelligence** - Redundancy + voting > individual quality

The Cluster Block creates emergent intelligence through evolutionary optimization of node populations working symbiotically.
