"""Minimal DELLM: Main entry point following minimal plan."""
import json
import sys
import atexit
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.node import Node
from minimal.orchestrator import Orchestrator
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="DELLM Minimal", version="0.1.0")


def load_topology(config_path: str = "config/topology.json"):
    """Load topology configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Topology config not found: {config_path}")
    with open(path, "r") as f:
        return json.load(f)


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str


def initialize_minimal_network():
    """Initialize minimal network from topology."""
    print("Loading topology...")
    topology = load_topology()
    
    # Create nodes
    print("Creating nodes...")
    nodes = {}
    for node_id, node_config in topology["nodes"].items():
        node = Node(
            node_id=node_id,
            model_path=node_config.get("model_path"),
            model_name=node_config.get("model_name", "tinyllama-1.1b")
        )
        node.fitness = node_config.get("fitness", 0.7)
        nodes[node_id] = node
    
    print(f"Created {len(nodes)} nodes")
    
    # Create orchestrator
    superllm_config = topology["superllm"]
    orchestrator = Orchestrator(
        nodes=nodes,
        synthesis_model_path=superllm_config.get("model_path")
    )
    
    print("\n" + "="*50)
    print("Minimal DELLM Network:")
    print(f"  Nodes: {len(nodes)}")
    print("="*50 + "\n")
    
    return orchestrator


# Global orchestrator
orchestrator = None


def initialize_orchestrator(orch):
    """Initialize global orchestrator."""
    global orchestrator
    orchestrator = orch


@app.post("/query")
async def process_query(request: QueryRequest):
    """Process query through minimal pipeline."""
    import time
    global orchestrator
    
    if orchestrator is None:
        return {"error": "Orchestrator not initialized"}
    
    print(f"\n{'='*70}")
    print(f"Received query: {request.query}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    try:
        result = await orchestrator.process_query(request.query)
        elapsed = time.time() - start_time
        print(f"\n[Server] Total processing time: {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[Server] Error after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "elapsed_time": elapsed}


@app.get("/health")
async def health_check():
    """Health check."""
    return {"status": "healthy", "orchestrator_initialized": orchestrator is not None}


def cleanup_on_exit():
    """Cleanup models on exit."""
    global orchestrator
    if orchestrator:
        try:
            orchestrator.close()
        except:
            pass
    from utils.llm_client import cleanup_all_models
    cleanup_all_models()


def main():
    """Main entry point."""
    global orchestrator
    print("Initializing Minimal DELLM...")
    
    # Register cleanup on exit
    atexit.register(cleanup_on_exit)
    
    orchestrator = initialize_minimal_network()
    initialize_orchestrator(orchestrator)
    
    print("Starting API server on http://localhost:8000")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cleanup_on_exit()


if __name__ == "__main__":
    main()

