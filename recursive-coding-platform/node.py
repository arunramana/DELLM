import asyncio
from llama_cpp import Llama
import docker
import uuid
import json
import time
from typing import Dict, List, Optional
import config
from dht_backend import create_dht

class Node:
    """Autonomous node that can generate code, delegate tasks, and test results"""
    
    def __init__(self, initial_peers: List = None):
        # Node identity
        self.node_id = str(uuid.uuid4())[:8]
        
        # Initialize DHT (hivemind on Linux/WSL, SharedState on Windows)
        self.dht = create_dht(initial_peers=initial_peers)
        
        # Load LLM
        print(f"[{self.node_id}] Loading model...")
        self.slm = Llama(
            model_path=config.MODEL_PATH,
            n_ctx=config.N_CTX,
            n_threads=config.N_THREADS,
            n_gpu_layers=config.N_GPU_LAYERS,
            verbose=False
        )
        
        # Docker client
        self.docker = docker.from_env()
        
        # State
        self.current_load = 0.0
        self.tasks_completed = 0
        self.specialties = []
        
        print(f"[{self.node_id}] Node initialized")
    
    async def start(self):
        """Start node services"""
        asyncio.create_task(self.heartbeat_loop())
        asyncio.create_task(self.task_listener_loop())
        print(f"[{self.node_id}] Node started and listening...")
    
    async def heartbeat_loop(self):
        """Regularly update status in DHT"""
        while True:
            try:
                await self.dht.astore(
                    key=f"node:{self.node_id}",
                    value={
                        "id": self.node_id,
                        "load": self.current_load,
                        "tasks_completed": self.tasks_completed,
                        "status": "available" if self.current_load < 0.8 else "busy",
                        "timestamp": time.time()
                    },
                    expiration_time=config.EXPIRATION_TIME
                )
                # Test: try to read back our own heartbeat to verify DHT is working
                test_read = await self.dht.aget(f"node:{self.node_id}")
                if test_read is None:
                    print(f"[{self.node_id}] [WARNING] Could not read back own heartbeat - DHT may not be working")
            except Exception as e:
                print(f"[{self.node_id}] Heartbeat error: {e}")
                import traceback
                print(f"[{self.node_id}] Traceback: {traceback.format_exc()}")
            
            await asyncio.sleep(config.HEARTBEAT_INTERVAL)
    
    async def task_listener_loop(self):
        """Check for tasks assigned to this node"""
        check_count = 0
        while True:
            try:
                # Check for tasks assigned to this node
                task = await self.get_assigned_task()
                
                if task:
                    print(f"[{self.node_id}] Found assigned task: {task.get('id', 'unknown')}")
                    asyncio.create_task(self.handle_task(task))
                else:
                    # Also check for broadcast tasks if we're not too busy (increased capacity)
                    if self.current_load < 2.0:
                        broadcast_task = await self.get_broadcast_task()
                        if broadcast_task:
                            print(f"[{self.node_id}] Found broadcast task: {broadcast_task.get('id', 'unknown')}")
                            asyncio.create_task(self.handle_task(broadcast_task))
                
                # Debug output every 15 checks (about once per 30 seconds)
                check_count += 1
                if check_count % 15 == 0:
                    backend_type = getattr(self.dht, "backend_type", "unknown")
                    # Check if there are any tasks waiting
                    if backend_type == "hivemind":
                        # For hivemind, check the queue directly
                        queue = await self.dht.aget("task_queue")
                        if queue and isinstance(queue, list) and queue:
                            print(f"[{self.node_id}] [DEBUG] Found {len(queue)} task(s) in queue: {queue}")
                        else:
                            print(f"[{self.node_id}] [DEBUG] No tasks in queue (queue={queue})")
                    else:
                        # SharedState path
                        all_tasks = self.dht.get_all_tasks("task:")
                        if all_tasks:
                            print(f"[{self.node_id}] [DEBUG] Found {len(all_tasks)} task(s) in queue: {list(all_tasks.keys())}")
                    # Also check node registration
                    nodes = self.dht.get_all_nodes()
                    print(f"[{self.node_id}] [DEBUG] Registered nodes: {list(nodes.keys())}")
                        
            except Exception as e:
                print(f"[{self.node_id}] Task listener error: {e}")
            
            await asyncio.sleep(2)
    
    async def get_assigned_task(self) -> Optional[Dict]:
        """Fetch task from shared state"""
        task_key = f"task:{self.node_id}:pending"
        
        try:
            task = await self.dht.aget(task_key)
            
            if task:
                await self.dht.adelete(task_key)
                return task
        except Exception as e:
            print(f"[{self.node_id}] Error fetching task: {e}")
        
        return None
    
    async def get_broadcast_task(self) -> Optional[Dict]:
        """Fetch a broadcast task if available.
        
        For SharedState backend we use an atomic pop_task (no race conditions).
        For hivemind backend we use a simple queue stored under 'task_queue'
        plus per-task keys 'task:broadcast:{task_id}'.
        """
        try:
            backend_type = getattr(self.dht, "backend_type", None)
            
            # Hivemind path: use a simple queue living at "task_queue"
            if backend_type == "hivemind":
                queue = await self.dht.aget("task_queue")
                
                # Debug logging
                if queue is None:
                    return None
                
                if not isinstance(queue, list):
                    print(f"[{self.node_id}] [DEBUG] task_queue is not a list: {type(queue)} = {queue}")
                    return None
                
                if not queue:
                    return None
                
                # Take the first task id from the queue
                task_id = queue[0]
                print(f"[{self.node_id}] [DEBUG] Found task_id '{task_id}' in queue (len={len(queue)})")
                
                # Remove this id from the queue (best-effort, not strictly atomic)
                new_queue = [tid for tid in queue if tid != task_id]
                # Store updated queue with a reasonably long expiration
                await self.dht.astore("task_queue", new_queue, expiration_time=3600)
                
                # Fetch the actual task payload
                task_key = f"task:broadcast:{task_id}"
                task = await self.dht.aget(task_key)
                if not task:
                    print(f"[{self.node_id}] [DEBUG] Task payload not found for key '{task_key}'")
                    return None
                
                print(f"[{self.node_id}] Picked up broadcast task: {task.get('id', 'unknown')}")
                return task
            
            # Default / SharedState path: use atomic pop_task to prevent races
            result = self.dht.pop_task("task:broadcast:")
            if result:
                task_key, task = result
                print(f"[{self.node_id}] Picked up broadcast task: {task.get('id', 'unknown')}")
                return task
            
        except Exception as e:
            import traceback
            print(f"[{self.node_id}] Error fetching broadcast task: {e}")
            print(f"[{self.node_id}] Traceback: {traceback.format_exc()}")
        
        return None
    
    async def handle_task(self, task: Dict):
        """Main task processing logic"""
        self.current_load += 0.3
        task_id = task.get('id', 'unknown')
        
        print(f"[{self.node_id}] Handling task: {task_id}")
        
        try:
            # Decide: Execute or Delegate?
            if self.should_execute(task):
                result = await self.execute_task(task)
            else:
                result = await self.delegate_task(task)
            
            # Store result
            await self.dht.astore(
                key=f"result:{task_id}",
                value=result,
                expiration_time=600
            )
            
            self.tasks_completed += 1
            print(f"[{self.node_id}] ✓ Completed task: {task_id}")
            
        except Exception as e:
            print(f"[{self.node_id}] ✗ Task failed: {task_id} - {e}")
            
            # Store failure
            await self.dht.astore(
                key=f"result:{task_id}",
                value={"success": False, "error": str(e)},
                expiration_time=600
            )
        
        finally:
            self.current_load -= 0.3
    
    def should_execute(self, task: Dict) -> bool:
        """Decide if this node should execute or delegate"""
        
        # If too busy, delegate
        if self.current_load > 0.7:
            print(f"[{self.node_id}] Too busy (load={self.current_load:.2f}), delegating")
            return False
        
        # Simple heuristic: if description is short, likely atomic
        desc = task.get('description', '')
        if len(desc.split()) < 10:
            print(f"[{self.node_id}] Short task description, executing directly")
            return True
        
        # Ask the model if this is atomic
        prompt = f"""Task: {task['description']}

Is this a single, atomic coding task that can be completed with one function or component?
Answer only YES or NO."""
        
        try:
            response = self.slm(
                prompt,
                max_tokens=10,
                temperature=0
            )
            
            answer = response['choices'][0]['text'].strip().upper()
            is_atomic = "YES" in answer or "ATOMIC" in answer
            
            print(f"[{self.node_id}] Task atomic? {is_atomic} (model response: {answer})")
            return is_atomic
            
        except Exception as e:
            print(f"[{self.node_id}] Decision error: {e}, defaulting to execute (safer)")
            # Default to execute rather than delegate to avoid delegation issues
            return True
    
    async def execute_task(self, task: Dict) -> Dict:
        """Generate code and test it"""
        
        print(f"[{self.node_id}] Executing task locally")
        
        # Build code generation prompt
        prompt = self.build_code_prompt(task)
        
        # Generate code
        for attempt in range(config.MAX_RETRIES):
            try:
                code_response = self.slm(
                    prompt,
                    max_tokens=2048,  # Increased for complex components
                    temperature=0.3
                )
                
                code = code_response['choices'][0]['text'].strip()
                
                # Clean up code - remove markdown code blocks if present
                if code.startswith('```'):
                    # Remove markdown code fence
                    lines = code.split('\n')
                    # Remove first line if it's a code fence
                    if lines[0].strip().startswith('```'):
                        lines = lines[1:]
                    # Remove last line if it's a code fence
                    if lines and lines[-1].strip().startswith('```'):
                        lines = lines[:-1]
                    code = '\n'.join(lines).strip()
                
                # Test if tests provided
                if task.get('tests'):
                    tests_passed = await self.test_code(code, task['tests'])
                    
                    if tests_passed:
                        return {
                            "success": True,
                            "code": code,
                            "node": self.node_id
                        }
                    else:
                        print(f"[{self.node_id}] Tests failed, retry {attempt + 1}/{config.MAX_RETRIES}")
                        continue
                else:
                    # No tests, return code
                    return {
                        "success": True,
                        "code": code,
                        "node": self.node_id
                    }
                    
            except Exception as e:
                print(f"[{self.node_id}] Execution error: {e}")
                
                if attempt == config.MAX_RETRIES - 1:
                    raise
        
        return {
            "success": False,
            "error": "Max retries exceeded",
            "node": self.node_id
        }
    
    def build_code_prompt(self, task: Dict) -> str:
        """Build context-aware code generation prompt"""
        
        mission = task.get('mission', {})
        context_chain = task.get('context_chain', [])
        description = task['description']
        
        # Build a better prompt for code generation
        if context_chain:
            context_text = self.format_context_chain(context_chain)
            prompt = f"""You are a coding assistant. Generate complete, working code for this task.

PARENT CONTEXT:
{context_text}

CURRENT TASK: {description}

IMPORTANT: Generate complete, working code. All functions referenced must be defined. All imports must be included. The code must be syntactically correct and runnable.

Return ONLY the code, no markdown, no explanations, no comments unless necessary for clarity.
"""
        else:
            prompt = f"""You are a coding assistant. Generate complete, working code for this task.

TASK: {description}

IMPORTANT: Generate complete, working code. All functions referenced must be defined. All imports must be included. The code must be syntactically correct and runnable.

Return ONLY the code, no markdown, no explanations, no comments unless necessary for clarity.
"""
        return prompt
    
    def format_context_chain(self, chain: List[Dict]) -> str:
        """Format context chain for prompt"""
        if not chain:
            return "Root task"
        
        formatted = []
        for i, ctx in enumerate(chain):
            formatted.append(f"Level {i}: {ctx.get('task', 'Unknown')}")
        
        return "\n".join(formatted)
    
    async def test_code(self, code: str, tests: List[Dict]) -> bool:
        """Execute code in Docker sandbox"""
        
        try:
            # Build test script
            test_script = f"""{code}

# Tests
{chr(10).join([f"assert {test['assertion']}" for test in tests])}
print("All tests passed")
"""
            
            # Run in Docker
            container = self.docker.containers.run(
                config.DOCKER_IMAGE,
                command=["python", "-c", test_script],
                detach=True,
                mem_limit=config.DOCKER_MEM_LIMIT,
                network_disabled=True,
                remove=True
            )
            
            result = container.wait(timeout=config.DOCKER_TIMEOUT)
            
            return result['StatusCode'] == 0
            
        except Exception as e:
            print(f"[{self.node_id}] Test execution error: {e}")
            return False
    
    async def delegate_task(self, task: Dict) -> Dict:
        """Break down and delegate to other nodes"""
        
        print(f"[{self.node_id}] Delegating task")
        
        # Decompose into subtasks
        subtasks = self.decompose_task(task)
        
        if not subtasks or len(subtasks) < 2:
            print(f"[{self.node_id}] Decomposition failed or insufficient subtasks, executing locally")
            return await self.execute_task(task)
        
        # Find available peers
        peers = self.discover_peers()
        
        if not peers:
            print(f"[{self.node_id}] No peers available, executing locally")
            return await self.execute_task(task)
        
        # Assign subtasks using broadcast mode for better distribution
        print(f"[{self.node_id}] Sending {len(subtasks)} subtasks to network (broadcast mode)")
        
        for subtask in subtasks:
            # Use broadcast mode so any available node can pick it up
            await self.send_task_as_broadcast(subtask)
        
        # Wait for results (with longer timeout)
        results = await self.wait_for_results([st['id'] for st in subtasks])
        
        # Check if we got enough results
        if len(results) < len(subtasks):
            print(f"[{self.node_id}] Only got {len(results)}/{len(subtasks)} subtask results, executing locally as fallback")
            return await self.execute_task(task)
        
        # Integrate
        integrated = self.integrate_results(task, results)
        
        # If integration failed, execute locally as fallback
        if not integrated.get('success'):
            print(f"[{self.node_id}] Integration failed, executing locally as fallback")
            return await self.execute_task(task)
        
        return integrated
    
    def decompose_task(self, task: Dict) -> List[Dict]:
        """Use SLM to break task into subtasks"""
        
        prompt = f"""Break this coding task into 2-3 atomic subtasks:

Task: {task['description']}

For each subtask, provide a brief description of what code it should generate.
Format as a simple list, one per line, starting with "1.", "2.", etc.
"""
        
        try:
            response = self.slm(
                prompt,
                max_tokens=256,
                temperature=0.5
            )
            
            text = response['choices'][0]['text'].strip()
            
            # Parse simple numbered list
            lines = text.split('\n')
            subtasks = []
            
            for i, line in enumerate(lines[:3]):  # Max 3 subtasks
                if line.strip():
                    # Remove numbering
                    desc = line.strip().lstrip('0123456789.)-').strip()
                    
                    if desc:
                        subtasks.append({
                            "id": f"{task['id']}.{i+1}",
                            "description": desc,
                            "mission": task.get('mission'),
                            "context_chain": task.get('context_chain', []) + [
                                {"task": task['description']}
                            ]
                        })
            
            return subtasks
            
        except Exception as e:
            print(f"[{self.node_id}] Decomposition error: {e}")
            return []
    
    def discover_peers(self) -> List[Dict]:
        """Find other nodes in the network"""
        
        peers = []
        
        try:
            # Get all nodes from shared state
            all_nodes = self.dht.get_all_nodes()
            
            for node_id, node_data in all_nodes.items():
                # Don't include ourselves
                if node_id != self.node_id:
                    peers.append({
                        "id": node_id,
                        "load": node_data.get("load", 1.0),
                        "status": node_data.get("status", "unknown"),
                        "tasks_completed": node_data.get("tasks_completed", 0)
                    })
            
        except Exception as e:
            print(f"[{self.node_id}] Peer discovery error: {e}")
        
        return peers
    
    def select_best_peer(self, peers: List[Dict]) -> Optional[Dict]:
        """Choose least loaded peer"""
        
        if not peers:
            return None
        
        return min(peers, key=lambda p: p.get('load', 1.0))
    
    async def send_task_to_peer(self, peer_id: str, task: Dict):
        """Store task in DHT for peer to pick up"""
        
        task_key = f"task:{peer_id}:pending"
        
        try:
            await self.dht.astore(
                key=task_key,
                value=task,
                expiration_time=300
            )
            
            print(f"[{self.node_id}] Sent task {task['id']} to {peer_id}")
            
        except Exception as e:
            print(f"[{self.node_id}] Error sending task: {e}")
    
    async def send_task_as_broadcast(self, task: Dict):
        """Store task as broadcast so any node can pick it up"""
        
        task_key = f"task:broadcast:{task['id']}"
        
        try:
            await self.dht.astore(
                key=task_key,
                value=task,
                expiration_time=600  # Longer expiration for subtasks
            )
            
            print(f"[{self.node_id}] Broadcast subtask {task['id']}: {task.get('description', '')[:50]}")
            
        except Exception as e:
            print(f"[{self.node_id}] Error broadcasting task: {e}")
    
    async def wait_for_results(self, task_ids: List[str]) -> Dict:
        """Wait for delegated tasks to complete"""
        
        results = {}
        start_time = time.time()
        last_log = 0
        
        print(f"[{self.node_id}] Waiting for {len(task_ids)} subtask results...")
        
        while len(results) < len(task_ids):
            # Check timeout (but make it longer)
            elapsed = time.time() - start_time
            if elapsed > config.TASK_TIMEOUT * 2:  # Double the timeout
                print(f"[{self.node_id}] Timeout waiting for results after {elapsed:.0f}s")
                print(f"[{self.node_id}] Got {len(results)}/{len(task_ids)} results so far")
                break
            
            # Log progress every 30 seconds
            if elapsed - last_log >= 30:
                print(f"[{self.node_id}] Still waiting... {len(results)}/{len(task_ids)} results received ({elapsed:.0f}s)")
                last_log = elapsed
            
            # Check for results
            for task_id in task_ids:
                if task_id not in results:
                    try:
                        result = await self.dht.aget(f"result:{task_id}")
                        
                        if result:
                            results[task_id] = result
                            print(f"[{self.node_id}] Received result for {task_id}: success={result.get('success', False)}")
                            
                    except Exception as e:
                        print(f"[{self.node_id}] Error fetching result for {task_id}: {e}")
            
            await asyncio.sleep(2)  # Check every 2 seconds
        
        return results
    
    def integrate_results(self, parent_task: Dict, results: Dict) -> Dict:
        """Combine subtask results"""
        
        successful_results = [
            r for r in results.values() 
            if r.get('success')
        ]
        
        failed_results = [
            (task_id, r.get('error', 'Unknown error'))
            for task_id, r in results.items()
            if not r.get('success')
        ]
        
        if failed_results:
            print(f"[{self.node_id}] Some subtasks failed: {failed_results}")
        
        if not successful_results:
            return {
                "success": False,
                "error": f"No successful subtasks. Failed: {failed_results}",
                "node": self.node_id
            }
        
        # Combine code from successful results
        code_parts = []
        for r in successful_results:
            code = r.get('code', '').strip()
            if code:
                code_parts.append(code)
        
        if not code_parts:
            return {
                "success": False,
                "error": "No code generated in successful subtasks",
                "node": self.node_id
            }
        
        combined_code = "\n\n".join(code_parts)
        
        print(f"[{self.node_id}] Integrated {len(successful_results)}/{len(results)} subtasks successfully")
        
        return {
            "success": True,
            "code": combined_code,
            "node": self.node_id,
            "integrated_from": [task_id for task_id, r in results.items() if r.get('success')]
        }
