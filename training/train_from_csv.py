"""Training script using CSV file."""
import csv
import asyncio
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from minimal.embedding_orchestrator import EmbeddingOrchestrator
from core.transformer_node import TransformerNode
from utils.config_loader import config


def get_device() -> str:
    """Auto-detect and return available device."""
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return "cuda"  # Enable GPU for sequential processing
    else:
        print("No GPU detected, using CPU")
        return "cpu"


def load_training_data(csv_path: str):
    """Load queries and answers from CSV."""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'query': row['Query'],
                'correct_answer': row['Answer']
            })
    return data


async def train_on_dataset(orchestrator: EmbeddingOrchestrator, training_data: list):
    """Train on dataset, one query at a time."""
    print(f"\nTraining on {len(training_data)} examples...\n")
    
    initial_fitness = {}
    for node_id, node in orchestrator.nodes.items():
        initial_fitness[node_id] = node.fitness
    
    # Track metrics
    quality_scores = []
    fitness_history = {node_id: [node.fitness] for node_id in orchestrator.nodes.keys()}
    
    for i, example in enumerate(training_data, 1):
        query = example['query']
        correct_answer = example['correct_answer']
        
        # Process query with correct answer (triggers training)
        result = await orchestrator.process_query(query, correct_answer=correct_answer)
        
        # Track quality
        quality = result.get('quality_score', 0.0)
        quality_scores.append(quality)
        
        # Track fitness updates
        fitness_updates = result.get('fitness_updates', {})
        for node_id, fitness in fitness_updates.items():
            fitness_history[node_id].append(fitness)
            initial_fitness[node_id] = fitness
        
        # Progress update every 5 examples
        if i % 5 == 0 or i == len(training_data):
            avg_quality = sum(quality_scores) / len(quality_scores)
            print(f"[{i}/{len(training_data)}] Avg Quality: {avg_quality:.3f} | ", end="")
            for node_id in sorted(orchestrator.nodes.keys()):
                current_fitness = initial_fitness.get(node_id, 0.7)
                print(f"{node_id}: {current_fitness:.3f}  ", end="")
            print()
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    # Accuracy metrics
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    high_quality_count = sum(1 for q in quality_scores if q >= 0.7)
    accuracy = high_quality_count / len(quality_scores) if quality_scores else 0.0
    
    print(f"\nAccuracy Metrics:")
    print(f"  Average Quality Score: {avg_quality:.3f}")
    print(f"  High Quality (>=0.7): {high_quality_count}/{len(quality_scores)} ({accuracy*100:.1f}%)")
    
    # Fitness changes
    print(f"\nFitness Changes:")
    for node_id in sorted(orchestrator.nodes.keys()):
        initial = fitness_history[node_id][0]
        final = fitness_history[node_id][-1]
        change = final - initial
        print(f"  {node_id}: {initial:.3f} -> {final:.3f} ({change:+.3f})")
    
    print("="*70)


def main():
    """Main training function."""
    import json
    import torch
    
    # Auto-detect device
    device = get_device()
    
    # Clear GPU cache if using GPU and reserve memory for external programs (like Cursor)
    if device == "cuda" and torch.cuda.is_available():
        # Force garbage collection and clear GPU cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        # Reserve ~1GB for external programs (Cursor, etc.)
        reserved_gb = 1.0
        available_gb = total_memory - reserved_gb
        # Set memory fraction to leave space for external programs
        torch.cuda.set_per_process_memory_fraction(available_gb / total_memory)
        print(f"GPU memory: {total_memory:.2f} GB total, reserving {reserved_gb:.2f} GB for external programs")
        print(f"Available for training: {available_gb:.2f} GB")
        
        # Print optimization settings
        keep_on_gpu = config.get('training', 'keep_models_on_gpu', default=True)
        use_mixed_precision = config.get('training', 'use_mixed_precision', default=True)
        use_torch_compile = config.get('training', 'use_torch_compile', default=False)
        print(f"\nGPU Optimizations:")
        print(f"  Keep models on GPU: {keep_on_gpu} (faster, uses more memory)")
        print(f"  Mixed Precision (FP16): {use_mixed_precision} (~2x faster, ~50% less memory)")
        print(f"  torch.compile(): {use_torch_compile} (faster inference, PyTorch 2.0+)")
        print()
    
    # Load topology
    topology_path = project_root / "config" / "topology.json"
    with open(topology_path, 'r') as f:
        topology = json.load(f)
    
    # Create nodes
    nodes = {}
    for node_id, node_config in topology["nodes"].items():
        model_name = node_config.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        node = TransformerNode(
            node_id=node_id,
            model_name=model_name,
            device=device  # Use auto-detected device
        )
        node.fitness = node_config.get("fitness", 0.7)
        nodes[node_id] = node
    
    # Create orchestrator
    superllm_config = topology["superllm"]
    embedding_model = superllm_config.get("embedding_model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    decoder_model = superllm_config.get("decoder_model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    orchestrator = EmbeddingOrchestrator(
        nodes=nodes,
        embedding_model_name=embedding_model,
        decoder_model_name=decoder_model,
        device=device  # Use auto-detected device
    )
    
    # Load training data
    csv_path = project_root / "training" / "queries_and_answers.csv"
    if not csv_path.exists():
        print(f"Error: Training CSV not found at {csv_path}")
        return
    
    training_data = load_training_data(str(csv_path))
    print(f"Loaded {len(training_data)} training examples from CSV")
    
    # Train
    try:
        asyncio.run(train_on_dataset(orchestrator, training_data))
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Cleanup
        orchestrator.close()
        print("\nCleanup completed")


if __name__ == "__main__":
    main()

