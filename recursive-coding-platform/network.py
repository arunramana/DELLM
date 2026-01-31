import asyncio
from node import Node
import config
import time
import json
import os

BOOTSTRAP_FILE = "hivemind_bootstrap.json"

async def launch_network():
    """Launch multiple nodes on this device"""
    
    print("=" * 60)
    print("RECURSIVE CODING PLATFORM - MVP")
    print("=" * 60)
    print(f"Launching {config.NUM_NODES} nodes...")
    print(f"Model: {config.MODEL_PATH}")
    print()
    
    nodes = []
    bootstrap_peers = []
    
    # Start first node (bootstrap)
    print("[NETWORK] Starting bootstrap node...")
    node1 = Node(initial_peers=[])
    await node1.start()
    nodes.append(node1)
    
    # Wait for DHT to initialize
    await asyncio.sleep(5)  # Increased wait time for hivemind to fully start
    
    # Get bootstrap peer address for hivemind (if using hivemind backend)
    backend_type = getattr(node1.dht, "backend_type", None)
    if backend_type == "hivemind":
        try:
            # Get the first node's DHT peer address
            dht_backend = node1.dht.backend
            # Try multiple methods to get peer address
            if hasattr(dht_backend, 'get_visible_maddrs'):
                maddrs = dht_backend.get_visible_maddrs()
                if maddrs:
                    bootstrap_peers = [str(maddr) for maddr in maddrs]
                    print(f"[NETWORK] Bootstrap peer address: {bootstrap_peers[0]}")
                    # Persist bootstrap peers for submit_task.py
                    try:
                        with open(BOOTSTRAP_FILE, "w", encoding="utf-8") as f:
                            json.dump({"peers": bootstrap_peers}, f)
                        print(f"[NETWORK] Wrote bootstrap peers to {BOOTSTRAP_FILE}")
                    except Exception as e:
                        print(f"[NETWORK] Warning: could not write {BOOTSTRAP_FILE}: {e}")
            elif hasattr(dht_backend, 'peer_id') and hasattr(dht_backend, 'port'):
                # Construct peer address manually
                peer_id = str(dht_backend.peer_id)
                port = getattr(dht_backend, 'port', 8000)
                bootstrap_peers = [f"/ip4/127.0.0.1/tcp/{port}/p2p/{peer_id}"]
                print(f"[NETWORK] Bootstrap peer address: {bootstrap_peers[0]}")
                try:
                    with open(BOOTSTRAP_FILE, "w", encoding="utf-8") as f:
                        json.dump({"peers": bootstrap_peers}, f)
                    print(f"[NETWORK] Wrote bootstrap peers to {BOOTSTRAP_FILE}")
                except Exception as e:
                    print(f"[NETWORK] Warning: could not write {BOOTSTRAP_FILE}: {e}")
            else:
                print("[NETWORK] Could not determine bootstrap peer address - nodes may auto-discover")
        except Exception as e:
            print(f"[NETWORK] Warning: Could not get bootstrap peer address: {e}")
            print("[NETWORK] Nodes will attempt to discover each other via local network")
            # Fallback: nodes will try to discover each other via local network
    
    # Start additional nodes with bootstrap peer address
    for i in range(config.NUM_NODES - 1):
        print(f"[NETWORK] Starting node {i+2}/{config.NUM_NODES}...")
        
        node = Node(initial_peers=bootstrap_peers)
        await node.start()
        nodes.append(node)
        
        await asyncio.sleep(2)
    
    print()
    print("=" * 60)
    print(f"✓ Network launched successfully with {len(nodes)} nodes")
    print("=" * 60)
    print()
    print("Node IDs:")
    for i, node in enumerate(nodes, 1):
        print(f"  {i}. {node.node_id}")
    
    print()
    print("Submit tasks using: python submit_task.py \"<your coding task>\"")
    print("Press Ctrl+C to stop the network")
    print()
    
    # Monitor network
    try:
        while True:
            await asyncio.sleep(10)
            
            # Print status
            print(f"\n[{time.strftime('%H:%M:%S')}] Network Status:")
            for node in nodes:
                status_emoji = "🟢" if node.current_load < 0.5 else "🟡" if node.current_load < 0.8 else "🔴"
                print(f"  {status_emoji} {node.node_id}: Load={node.current_load:.1f}, Tasks={node.tasks_completed}")
    
    except KeyboardInterrupt:
        print("\n\nShutting down network...")
        print("✓ Network stopped")

if __name__ == "__main__":
    asyncio.run(launch_network())
