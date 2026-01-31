"""
Platform-aware DHT backend selector.
Uses hivemind DHT on Linux/WSL, falls back to file-based SharedState on Windows.
"""
import os
import platform
import asyncio
import inspect
import time
from typing import Dict, Optional, Any, Tuple

def is_wsl():
    """Check if running in WSL"""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower() or 'wsl' in f.read().lower()
    except:
        return False

def is_linux():
    """Check if running on Linux (including WSL)"""
    return platform.system() == 'Linux' or is_wsl()

def get_dht_backend():
    """
    Returns the appropriate DHT backend based on platform.
    Returns 'hivemind' for Linux/WSL, 'file' for Windows.
    """
    if is_linux():
        try:
            import hivemind
            return 'hivemind'
        except ImportError:
            print("[WARNING] hivemind not installed, falling back to file-based storage")
            return 'file'
    else:
        return 'file'

class DHTWrapper:
    """Unified wrapper for both hivemind DHT and SharedState.
    
    backend_type is an explicit string: either "hivemind" or "shared_state".
    We pass this in from create_dht to avoid fragile introspection.
    """
    
    def __init__(self, backend_instance, backend_type: str):
        self.backend = backend_instance
        # Expected values: "hivemind" or "shared_state"
        self.backend_type = backend_type

    async def astore(self, key: str, value: Any, expiration_time: float = 300):
        """Async store (required for hivemind)."""
        if self.backend_type == "hivemind":
            # hivemind.DHT.store may be sync or async depending on version
            # hivemind expects an absolute UNIX timestamp, not a TTL seconds value
            expiration = time.time() + float(expiration_time)
            res = self.backend.store(key, value, expiration_time=expiration)
            if inspect.isawaitable(res):
                await res
        else:
            # SharedState is sync; run inline
            self.backend.store(key, value, expiration_time)

    async def aget(self, key: str) -> Optional[Any]:
        """Async get (required for hivemind)."""
        if self.backend_type == "hivemind":
            # hivemind.DHT.get may be sync or async depending on version
            res = self.backend.get(key, latest=True)
            if inspect.isawaitable(res):
                res = await res
            if res is None:
                return None

            # Different hivemind versions return different shapes:
            # - ValueWithExpiration (has .value)
            # - dict(peer_id -> ValueWithExpiration)
            # - raw value
            if hasattr(res, "value"):
                return res.value
            if isinstance(res, dict) and res:
                first = next(iter(res.values()))
                return getattr(first, "value", first)
            return res
        else:
            return self.backend.get(key)

    async def adelete(self, key: str):
        """Async delete helper."""
        if self.backend_type == "hivemind":
            # best-effort delete: overwrite with short expiration
            expiration = time.time() + 1e-3
            res = self.backend.store(key, None, expiration_time=expiration)
            if inspect.isawaitable(res):
                await res
        else:
            self.backend.delete(key)
    
    def store(self, key: str, value: Any, expiration_time: float = 300):
        """Synchronous store (SharedState only). For hivemind, use astore()."""
        if self.backend_type == "hivemind":
            raise RuntimeError("hivemind backend requires await dht.astore(...)")
        self.backend.store(key, value, expiration_time)
    
    def get(self, key: str) -> Optional[Any]:
        """Synchronous get (SharedState only). For hivemind, use aget()."""
        if self.backend_type == "hivemind":
            raise RuntimeError("hivemind backend requires await dht.aget(...)")
        return self.backend.get(key)
    
    def delete(self, key: str):
        """Synchronous delete (SharedState only). For hivemind, use adelete()."""
        if self.backend_type == "hivemind":
            raise RuntimeError("hivemind backend requires await dht.adelete(...)")
        self.backend.delete(key)
    
    def get_all_nodes(self) -> Dict[str, Dict]:
        """Get all node status entries"""
        if self.backend_type == 'hivemind':
            # For hivemind, we can't easily list all keys
            # Return empty dict - nodes will be discovered through heartbeats
            # In practice, nodes discover each other through the DHT network
            return {}
        else:
            return self.backend.get_all_nodes()
    
    def get_all_tasks(self, prefix: str = "task:") -> Dict[str, Dict]:
        """Get all tasks with given prefix"""
        if self.backend_type == 'hivemind':
            # hivemind doesn't support listing all keys easily
            # Return empty dict - we'll use pop_task instead
            return {}
        else:
            return self.backend.get_all_tasks(prefix)
    
    def pop_task(self, prefix: str = "task:") -> Optional[Tuple[str, Dict]]:
        """
        Atomically get and delete the first task matching prefix.
        Returns (task_key, task_value) or None if no tasks available.
        """
        if self.backend_type == 'hivemind':
            # For hivemind, we can't easily list all keys atomically
            # We'll need to use a different approach - try to get and delete in a loop
            # For broadcast tasks, we can try common patterns
            # This is a limitation of hivemind - it's designed for key-value lookups, not listing
            # For now, return None and let nodes use a different discovery mechanism
            # In practice, nodes would need to know task IDs or use a different pattern
            return None
        else:
            result = self.backend.pop_task(prefix)
            return result

def create_dht(initial_peers=None):
    """
    Create and return a DHT wrapper instance based on platform.
    Returns a DHTWrapper that works with both hivemind and SharedState.
    
    For hivemind: initial_peers should be a list of peer addresses (e.g., ["/ip4/127.0.0.1/tcp/8000/p2p/..."]).
    If None or empty, the DHT will try to discover peers via local network.
    """
    backend = get_dht_backend()
    
    if backend == 'hivemind':
        import hivemind
        print("[DHT] Using hivemind DHT backend (Linux/WSL detected)")
        
        # Convert initial_peers to the format hivemind expects
        peers = initial_peers or []
        
        dht_instance = hivemind.DHT(
            initial_peers=peers,
            start=True
        )
        # Explicitly mark this wrapper as hivemind-backed
        return DHTWrapper(dht_instance, backend_type="hivemind")
    else:
        from shared_state import SharedState
        print("[DHT] Using file-based SharedState backend (Windows/SharedState detected)")
        return DHTWrapper(SharedState.get_instance(), backend_type="shared_state")
