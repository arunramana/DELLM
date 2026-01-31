"""
Windows-compatible shared state for node communication.
Replaces hivemind DHT which doesn't work on Windows due to add_reader limitations.
Uses file-based storage to work across processes.
"""
import threading
import time
import json
import os
from typing import Dict, Optional, Any
from collections import defaultdict

class SharedState:
    """Process-safe shared state for node communication using file-based storage"""
    
    _instance = None
    _lock = threading.Lock()
    _file_path = "shared_state.json"
    _lock_file_path = "shared_state.lock"
    
    def __init__(self):
        self._lock = threading.RLock()
        self._cleanup_interval = 30  # Clean expired entries every 30 seconds
        self._last_cleanup = time.time()
        # Ensure file exists
        if not os.path.exists(self._file_path):
            self._save_data({}, {})
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _acquire_file_lock(self, timeout=5):
        """Acquire file lock for cross-process synchronization"""
        try:
            # Simple file-based locking using existence check
            # Try to create lock file, retry if it exists
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    # Try to create lock file exclusively
                    if os.name == 'nt':
                        # Windows: use exclusive create
                        lock_file = open(self._lock_file_path, 'x')
                    else:
                        # Unix: use fcntl
                        import fcntl
                        lock_file = open(self._lock_file_path, 'w')
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return lock_file
                except (FileExistsError, IOError, BlockingIOError):
                    # Lock file exists, wait and retry
                    time.sleep(0.05)
                    continue
            # Timeout - proceed anyway (may cause minor race conditions)
            return None
        except Exception as e:
            # Fallback: proceed without locking
            return None
    
    def _release_file_lock(self, lock_file):
        """Release file lock"""
        if lock_file:
            try:
                lock_file.close()
                if os.path.exists(self._lock_file_path):
                    os.remove(self._lock_file_path)
            except:
                pass
    
    def _load_data(self):
        """Load data from file"""
        try:
            if os.path.exists(self._file_path):
                with open(self._file_path, 'r') as f:
                    data = json.load(f)
                    return data.get('data', {}), data.get('expiration', {})
        except Exception as e:
            print(f"[SharedState] Error loading data: {e}")
        return {}, {}
    
    def _save_data(self, data, expiration):
        """Save data to file"""
        try:
            with open(self._file_path, 'w') as f:
                json.dump({
                    'data': data,
                    'expiration': expiration
                }, f)
        except Exception as e:
            print(f"[SharedState] Error saving data: {e}")
    
    def _cleanup_expired(self, data, expiration):
        """Remove expired entries"""
        now = time.time()
        expired_keys = [
            key for key, exp_time in expiration.items()
            if exp_time < now
        ]
        for key in expired_keys:
            data.pop(key, None)
            expiration.pop(key, None)
        return data, expiration
    
    def store(self, key: str, value: Any, expiration_time: float = 300):
        """Store a value with expiration"""
        lock_file = self._acquire_file_lock()
        try:
            data, expiration = self._load_data()
            data[key] = value
            expiration[key] = time.time() + expiration_time
            data, expiration = self._cleanup_expired(data, expiration)
            self._save_data(data, expiration)
        finally:
            self._release_file_lock(lock_file)
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value if it exists and hasn't expired"""
        lock_file = self._acquire_file_lock()
        try:
            data, expiration = self._load_data()
            data, expiration = self._cleanup_expired(data, expiration)
            
            if key in expiration:
                if expiration[key] < time.time():
                    # Expired
                    data.pop(key, None)
                    expiration.pop(key, None)
                    self._save_data(data, expiration)
                    return None
            
            result = data.get(key)
            # Save cleaned data
            self._save_data(data, expiration)
            return result
        finally:
            self._release_file_lock(lock_file)
    
    def delete(self, key: str):
        """Delete a key"""
        lock_file = self._acquire_file_lock()
        try:
            data, expiration = self._load_data()
            data.pop(key, None)
            expiration.pop(key, None)
            self._save_data(data, expiration)
        finally:
            self._release_file_lock(lock_file)
    
    def get_all_nodes(self) -> Dict[str, Dict]:
        """Get all node status entries"""
        lock_file = self._acquire_file_lock()
        try:
            data, expiration = self._load_data()
            data, expiration = self._cleanup_expired(data, expiration)
            nodes = {}
            for key, value in data.items():
                if key.startswith("node:") and isinstance(value, dict):
                    node_id = key.split(":", 1)[1]
                    nodes[node_id] = value
            self._save_data(data, expiration)
            return nodes
        finally:
            self._release_file_lock(lock_file)
    
    def get_all_tasks(self, prefix: str = "task:") -> Dict[str, Dict]:
        """Get all tasks with given prefix"""
        lock_file = self._acquire_file_lock()
        try:
            data, expiration = self._load_data()
            data, expiration = self._cleanup_expired(data, expiration)
            tasks = {}
            for key, value in data.items():
                if key.startswith(prefix) and isinstance(value, dict):
                    tasks[key] = value
            self._save_data(data, expiration)
            return tasks
        finally:
            self._release_file_lock(lock_file)
    
    def pop_task(self, prefix: str = "task:") -> Optional[tuple]:
        """
        Atomically get and delete the first task matching prefix.
        Returns (task_key, task_value) or None if no tasks available.
        This prevents race conditions where multiple nodes pick the same task.
        """
        lock_file = self._acquire_file_lock()
        try:
            data, expiration = self._load_data()
            data, expiration = self._cleanup_expired(data, expiration)
            
            # Find first matching task
            for key, value in data.items():
                if key.startswith(prefix) and isinstance(value, dict):
                    # Check if expired
                    if key in expiration and expiration[key] < time.time():
                        continue
                    # Atomically remove it
                    task_value = data.pop(key)
                    expiration.pop(key, None)
                    self._save_data(data, expiration)
                    return (key, task_value)
            
            # No tasks found
            self._save_data(data, expiration)
            return None
        finally:
            self._release_file_lock(lock_file)
