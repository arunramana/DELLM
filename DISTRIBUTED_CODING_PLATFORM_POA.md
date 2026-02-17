# DISTRIBUTED CODING PLATFORM - MVP IMPLEMENTATION PLAN

## QUICK START FOR CLAUDE CODE

**Repository Status**: 
- Directory `node/models/` exists with `qwen2.5-1.5b-instruct-q8_0.gguf` (1.1GB model file)
- Ready for implementation

**To begin with Claude Code**:
1. Add this POA document to your repository
2. Connect Claude Code to the repository
3. Start with: "Implement the distributed coding platform following this POA. Begin with step 1: create remaining project structure and implement shared/models.py"

**Key Changes from Original PDF**:
- Worker directory renamed to `node/` for clarity
- Model already downloaded at `node/models/qwen2.5-1.5b-instruct-q8_0.gguf`
- Main worker script is `node/worker.py` (not `node/node.py`)

---

## OVERVIEW

Build a simplified peer-to-peer distributed coding platform using:
- Hivemind for P2P networking via DHT
- Qwen 2.5-1.5B-Instruct as worker node LLM (1.1GB model)
- Claude API for orchestrator decomposition (temporary, will self-host later)
- Docker for sandboxed code execution
- GitPython for version control

This MVP validates core DELLM concepts before adding evolutionary mechanisms.

---

## ARCHITECTURE

```
User Query
    ↓
Orchestrator (Your Machine)
  - Receives query
  - Calls Claude API for task breakdown
  - Distributes tasks via Hivemind DHT
  - Collects results
  - Merges code to git repo
    ↓
Hivemind DHT Network (Distributed Hash Table)
    ↓
Worker Nodes (3-5 Peer Machines)
  - Poll DHT for tasks
  - Run Qwen 2.5-1.5B for code generation
  - Execute/test code in Docker
  - Return results to DHT
```

---

## PROJECT STRUCTURE

```
dellm-mvp/
├── orchestrator/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── task_decomposer.py      # Claude API calls
│   ├── task_distributor.py     # DHT task publishing
│   ├── result_collector.py     # DHT result gathering
│   ├── git_manager.py          # Git operations
│   └── config.py               # Configuration
│
├── node/
│   ├── models/
│   │   └── qwen2.5-1.5b-instruct-q8_0.gguf  # Pre-downloaded LLM (1.1GB)
│   ├── __init__.py
│   ├── worker.py               # Worker main loop
│   ├── task_executor.py        # Qwen inference
│   ├── docker_runner.py        # Code execution
│   ├── hivemind_client.py      # DHT client
│   └── config.py               # Worker config
│
├── shared/
│   ├── __init__.py
│   ├── models.py               # Task/Result dataclasses
│   ├── hivemind_utils.py       # DHT helpers
│   └── prompts.py              # LLM prompt templates
│
├── docker/
│   └── Dockerfile.sandbox      # Isolated execution env
│
├── tests/
│   ├── test_orchestrator.py
│   ├── test_worker.py
│   └── test_integration.py
│
├── scripts/
│   ├── setup_bootstrap.sh      # Initialize bootstrap node
│   └── start_worker.sh         # Worker startup script
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## WORKFLOW SPECIFICATION

### Step 1: User Submits Query
```
User provides:
- Coding task description
- Target git repository path

Example: "Create a Flask REST API with user authentication and SQLite database"
```

### Step 2: Task Decomposition (Orchestrator)
```
Component: orchestrator/task_decomposer.py

Input: User query string
Process:
  1. Format prompt for Claude API (see Query 5.1 from PDF)
  2. Call Claude API with specific instruction to break down into subtasks
  3. Parse JSON response into Task objects

Output: List[Task]
  Each Task contains:
  - id: int
  - description: str
  - file: str (target filename)
  - dependencies: List[int]
  - input_output_spec: str
  - directory: str
  - context: str

Query Format (from PDF Section 5.1):
"Given {user_query}:
I need you to understand the user query and break it down into subproblems or 
subtasks that can be solved by a network of SLMs to solve the user's query.

Create a concise and clear plan of action to break down the user's requirement 
into smaller tasks. The task should include:
- inputs and outputs required
- directory and repository to the file where the code has to be written to
- context of the task it has to solve
- Test each piece of code generated in each node

Ensure the tasks are completed in the correct order in p.o.a.
The p.o.a should have the list of all the tasks.

Return as valid JSON array with this structure:
[{
  "id": 1,
  "description": "Create models.py with User table",
  "file": "models.py",
  "dependencies": [],
  "input_output_spec": "Input: None, Output: SQLAlchemy User model class",
  "directory": "app/",
  "context": "Database models for user authentication"
}]"
```

### Step 3: Task Distribution (Orchestrator)
```
Component: orchestrator/task_distributor.py

Input: List[Task]
Process:
  1. Connect to Hivemind DHT
  2. For each task:
     - Serialize Task to dict
     - Store in DHT with key "task:{task_id}"
     - Set expiration time (300 seconds)
  3. Log distribution confirmation

DHT Key Format:
  - Tasks: "task:{task_id}"
  - Results: "result:{task_id}"
  - Status: "status:{task_id}"
```

### Step 4: Task Discovery (Worker Nodes)
```
Component: node/worker.py

Process (infinite loop):
  1. Connect to Hivemind DHT
  2. Poll DHT for available tasks
     - Query keys matching "task:*"
     - Filter for unclaimed tasks (no corresponding "status:claimed:{task_id}")
  3. Claim task by writing "status:claimed:{task_id}" with worker_id
  4. Download task data from DHT
  5. Execute task (Step 5)
  6. Upload result (Step 6)
  7. Sleep 2 seconds, repeat
```

### Step 5: Code Generation (Worker Node)
```
Component: node/task_executor.py

Input: Task object
Process:
  1. Check if task can be completed by this SLM (Query 5.2 from PDF)
  2. If too complex, divide into subtasks (Query 5.3 from PDF)
  3. Generate code using Qwen (Query 5.6 from PDF)
  4. Validate generated code syntax
  5. Return generated code

Query 5.2 - Complexity Check:
"Given:
- User query: {user_query}
- Plan of action: {p.o.a}
- Task: {task}
- Input/Output: {input_output}
- Directory: {directory}

Can the given task be completely solved by this SLM? 
Answer ONLY 'yes' or 'no'."

Query 5.3 - Task Division (if 5.2 returns 'no'):
"Given:
- User query: {user_query}
- Plan of action: {p.o.a}
- Task: {task}
- Input/Output: {input_output}
- Directory: {directory}

Divide the task into 2-4 subtasks so they can be solved by other nodes.
Try to minimize the number of subtasks needed to solve this task.

Return as JSON array:
[{
  "subtask_description": "...",
  "subtask_input_output": "...",
  "subtask_order": 1
}]"

Query 5.6 - Code Generation:
"Given:
- User query: {user_query}
- Plan of action: {p.o.a}
- Task: {task}
- Input/Output: {input_output}
- Directory: {directory}

Solve the task:
1. Ensure inputs and outputs match properly
2. Write test cases to test this code
3. Write comments to document and reason
4. Generate production-ready Python code

Output ONLY the code, no explanations, no markdown."

Token Limit Compliance:
- Max input to Qwen: 32,000 tokens
- Estimated prompt size: ~500-1000 tokens
- Reserve 2048 tokens for output
- Max safe input: 29,000 tokens
- If user_query + p.o.a exceeds 28k tokens, truncate p.o.a to summary
```

### Step 6: Code Testing (Worker Node)
```
Component: node/docker_runner.py

Input: Generated code string, Task object
Process:
  1. Create temporary directory
  2. Write code to file
  3. Write test file (from code generation step)
  4. Launch Docker container:
     - Image: python:3.11-slim
     - Mount temp directory as /code
     - Run: python -m pytest /code/
     - Capture stdout, stderr
     - Get exit code
  5. Parse test results
  6. Return TestResult object

TestResult fields:
  - success: bool (exit code == 0)
  - stdout: str
  - stderr: str
  - tests_passed: int
  - tests_failed: int
  - execution_time: float
```

### Step 7: Result Upload (Worker Node)
```
Component: node/worker.py

Input: Task object, Generated code, TestResult
Process:
  1. Create Result object:
     - task_id
     - code (generated code string)
     - test_passed (bool)
     - test_logs (stdout + stderr)
     - execution_time
     - worker_id
  2. Serialize Result to dict
  3. Upload to DHT with key "result:{task_id}"
  4. Remove "status:claimed:{task_id}" from DHT
  5. Log completion
```

### Step 8: Result Collection (Orchestrator)
```
Component: orchestrator/result_collector.py

Input: List[Task], timeout=120 seconds
Process:
  1. Initialize results dict
  2. Start timer
  3. While not all results collected and not timed out:
     - For each task:
       - Query DHT for "result:{task_id}"
       - If exists, parse and store in results dict
     - Sleep 1 second
  4. If timeout, raise TimeoutError with partial results
  5. Return Dict[task_id, Result]
```

### Step 9: Git Integration (Orchestrator)
```
Component: orchestrator/git_manager.py

Input: Dict[task_id, Result], repo_path
Process:
  1. Check if repo_path exists
  2. If not, git init repo_path
  3. For each result (in dependency order):
     - Create directory structure if needed
     - Write code to file: repo_path + directory + file
     - git add file
  4. git commit -m "Auto-generated by DELLM: {user_query}"
  5. Return commit hash
```

### Step 10: User Notification (Orchestrator)
```
Component: orchestrator/main.py

Input: Results, repo_path, commit_hash
Process:
  1. Generate summary using Query 5.7 from PDF
  2. Return summary to user

Query 5.7 - Final Message:
"Given:
- User query: {user_query}
- Plan of action: {p.o.a}

Now that all the code has been generated, tell the user what has been done 
to solve their query and how to run and test the code. Also mention the 
repository where the code was modified.

Keep response under 200 words."

Output Format:
Task completed!

Generated {N} files:
- {file1} (passed tests)
- {file2} (passed tests)
- {file3} (FAILED tests - see logs)

Location: {repo_path}
Commit: {commit_hash}

To run:
cd {repo_path}
python -m pytest

Test Results Summary:
- {X} tests passed
- {Y} tests failed
```

---

## DATA MODELS

### shared/models.py

```python
from dataclasses import dataclass, asdict
from typing import List, Optional
import json

@dataclass
class Task:
    id: int
    description: str
    file: str
    dependencies: List[int]
    input_output_spec: str
    directory: str
    context: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @staticmethod
    def from_dict(data: dict) -> 'Task':
        return Task(**data)
    
    @staticmethod
    def from_json(json_str: str) -> 'Task':
        return Task.from_dict(json.loads(json_str))

@dataclass
class TestResult:
    success: bool
    stdout: str
    stderr: str
    tests_passed: int
    tests_failed: int
    execution_time: float
    
    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class Result:
    task_id: int
    code: str
    test_passed: bool
    test_logs: str
    execution_time: float
    worker_id: str
    timestamp: float
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @staticmethod
    def from_dict(data: dict) -> 'Result':
        return Result(**data)
    
    @staticmethod
    def from_json(json_str: str) -> 'Result':
        return Result.from_dict(json.loads(json_str))
```

---

## CONFIGURATION FILES

### orchestrator/config.py

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Claude API
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS = 4096

# Hivemind
BOOTSTRAP_PEERS = os.getenv("HIVEMIND_BOOTSTRAP_PEERS", "").split(",")
DHT_PORT = int(os.getenv("DHT_PORT", "12345"))

# Timeouts
TASK_TIMEOUT = 120  # seconds
RESULT_COLLECTION_POLL_INTERVAL = 1  # seconds

# Git
DEFAULT_COMMIT_MESSAGE = "Auto-generated by DELLM distributed coding platform"
```

### node/config.py

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Model
MODEL_PATH = os.getenv("MODEL_PATH", "node/models/qwen2.5-1.5b-instruct-q8_0.gguf")
MODEL_N_CTX = 32000  # Max context window
MODEL_N_THREADS = int(os.getenv("MODEL_THREADS", "4"))
MODEL_TEMPERATURE = 0.2
MODEL_MAX_TOKENS = 2048

# Hivemind
BOOTSTRAP_PEERS = os.getenv("HIVEMIND_BOOTSTRAP_PEERS", "").split(",")
DHT_PORT = int(os.getenv("DHT_PORT", "12345"))
WORKER_ID = os.getenv("WORKER_ID", f"worker_{os.getpid()}")

# Docker
DOCKER_IMAGE = "python:3.11-slim"
DOCKER_TIMEOUT = 30  # seconds

# Polling
TASK_POLL_INTERVAL = 2  # seconds
```

### .env.example

```
# Orchestrator
ANTHROPIC_API_KEY=your_api_key_here
HIVEMIND_BOOTSTRAP_PEERS=/ip4/127.0.0.1/tcp/12345/p2p/QmBootstrapPeerID

# Worker Node
MODEL_PATH=node/models/qwen2.5-1.5b-instruct-q8_0.gguf
MODEL_THREADS=4
WORKER_ID=worker_01
DHT_PORT=12345
```

---

## DEPENDENCIES

### requirements.txt

```
# P2P Networking
hivemind==1.1.10

# LLM Inference
llama-cpp-python==0.2.90

# Orchestrator LLM
anthropic==0.40.0

# Code Execution
docker==7.1.0

# Git
GitPython==3.1.43

# Utilities
python-dotenv==1.0.1
pydantic==2.9.2
pytest==8.3.3

# Development
black==24.8.0
flake8==7.1.1
mypy==1.11.2
```

---

## CRITICAL IMPLEMENTATION NOTES

### Token Limit Management
```
Qwen 2.5-1.5B max input: 32,000 tokens
Safety margin: 3,000 tokens (for overhead)
Usable input: 29,000 tokens

Prompt components (estimated):
- System prompt: 200 tokens
- Query 5.2 (complexity check): 150 tokens
- Query 5.3 (task division): 200 tokens
- Query 5.6 (code gen): 500 tokens
- User query: variable (200-5000 tokens)
- P.o.a: variable (500-10000 tokens)
- Task details: variable (100-1000 tokens)

Token management strategy:
1. Calculate total prompt length before sending
2. If > 29,000 tokens:
   - Summarize p.o.a to 500 tokens max
   - Truncate user_query if needed (rare)
3. Always reserve 2048 tokens for output
```

### Hivemind DHT Usage
```
Key naming convention:
- task:{task_id} = Task object (JSON serialized)
- result:{task_id} = Result object (JSON serialized)
- status:claimed:{task_id} = worker_id (string)
- status:completed:{task_id} = timestamp (float)

Expiration times:
- Tasks: 300 seconds (5 minutes)
- Results: 600 seconds (10 minutes)
- Status markers: 300 seconds

Conflict resolution:
- First worker to write status:claimed wins
- If status exists, skip task
- Use atomic DHT operations
```

### Docker Execution Safety
```
Security measures:
1. Use python:3.11-slim (minimal attack surface)
2. Mount code directory as read-only
3. Set memory limit: 512MB
4. Set CPU limit: 1 core
5. Set timeout: 30 seconds
6. No network access (--network none)
7. Drop all capabilities
8. Run as non-root user

Example Docker command:
docker run \
  --rm \
  --memory=512m \
  --cpus=1 \
  --network=none \
  --user=1000:1000 \
  --cap-drop=ALL \
  -v /tmp/code:/code:ro \
  python:3.11-slim \
  timeout 30s python /code/test.py
```

### Git Merge Strategy
```
Dependency-aware merging:
1. Sort tasks by dependencies (topological sort)
2. Write files in dependency order
3. If file exists, overwrite (last write wins for MVP)
4. Commit after all files written
5. Future: implement proper merge conflict resolution

Example dependency resolution:
Task 1: models.py (deps: [])
Task 2: auth.py (deps: [1])
Task 3: routes.py (deps: [1, 2])
Task 4: tests.py (deps: [3])

Write order: 1 → 2 → 3 → 4
```

---

## DEPLOYMENT SEQUENCE

### Phase 1: Environment Setup
```
Step 1.1: Create project structure
mkdir -p dellm-mvp/{orchestrator,node,shared,docker,tests,scripts}
# Note: node/models/ already exists with qwen2.5-1.5b-instruct-q8_0.gguf

Step 1.2: Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Step 1.3: Configure environment
cp .env.example .env
# Edit .env with ANTHROPIC_API_KEY

Step 1.4: Build Docker sandbox
docker build -f docker/Dockerfile.sandbox -t dellm-sandbox:latest .
```

### Phase 2: Bootstrap Node
```
Step 2.1: Start Hivemind bootstrap peer
python -m hivemind.dht --port 12345 --bootstrap

Step 2.2: Record peer ID
# Save the peer ID (e.g., /ip4/127.0.0.1/tcp/12345/p2p/QmXXXXX)
# Update HIVEMIND_BOOTSTRAP_PEERS in .env
```

### Phase 3: Worker Deployment
```
Step 3.1: Start worker node 1 (Terminal 1)
export WORKER_ID=worker_01
python -m node.worker

Step 3.2: Start worker node 2 (Terminal 2)
export WORKER_ID=worker_02
python -m node.worker

Step 3.3: Start worker node 3 (Terminal 3)
export WORKER_ID=worker_03
python -m node.worker

Step 3.4: Verify workers connected
# Check DHT peer count
# Should see 3 workers + 1 bootstrap = 4 peers
```

### Phase 4: Test Execution
```
Step 4.1: Simple test
python -m orchestrator.main \
  --query "Create a simple Python function to calculate fibonacci" \
  --repo ./test-repo

Step 4.2: Medium complexity test
python -m orchestrator.main \
  --query "Create a Flask hello world API with a single /hello endpoint" \
  --repo ./test-repo

Step 4.3: Complex test
python -m orchestrator.main \
  --query "Create a Flask REST API with user auth and SQLite database" \
  --repo ./test-repo
```

---

## SUCCESS METRICS

### MVP Completion Criteria
```
1. Task Decomposition
   - Claude API successfully breaks queries into 3-6 tasks
   - Tasks have proper dependencies specified
   - 90% success rate

2. P2P Communication
   - Workers discover tasks via DHT
   - Results returned to DHT within timeout
   - 95% reliability

3. Code Generation
   - Qwen generates syntactically valid Python
   - 80% of generated code passes basic tests
   - No token limit errors

4. Code Execution
   - Docker containers run without crashes
   - Test results captured correctly
   - 100% isolation maintained

5. Git Integration
   - Files written to correct directories
   - Commits successful
   - Dependency order respected

6. End-to-End
   - User receives working code in git repo
   - Code runs without manual fixes
   - 70% success rate for medium complexity tasks
```

### Performance Benchmarks
```
Latency targets:
- Task decomposition: < 5 seconds
- Task distribution: < 1 second
- Code generation (per task): < 30 seconds
- Docker testing: < 10 seconds
- Result collection: < 5 seconds
- Total (3 parallel tasks): < 45 seconds

Throughput targets:
- 3 workers should handle 6 tasks/minute
- Linear scaling: 6 workers → 12 tasks/minute

Resource usage:
- Orchestrator: < 500MB RAM, < 10% CPU
- Worker: < 2GB RAM (model loaded), < 50% CPU
- Docker: < 512MB RAM per container
```

---

## FILE IMPLEMENTATION CHECKLIST

Use this checklist to implement each file:

### Orchestrator Files
- [ ] orchestrator/__init__.py
- [ ] orchestrator/config.py (load env vars)
- [ ] orchestrator/task_decomposer.py (Claude API integration, Query 5.1)
- [ ] orchestrator/task_distributor.py (DHT publishing)
- [ ] orchestrator/result_collector.py (DHT polling)
- [ ] orchestrator/git_manager.py (git operations)
- [ ] orchestrator/main.py (CLI entry point)

### Node Files (Worker)
- [ ] node/__init__.py
- [ ] node/config.py (load env vars)
- [ ] node/hivemind_client.py (DHT client)
- [ ] node/task_executor.py (Qwen inference, Queries 5.2, 5.3, 5.6)
- [ ] node/docker_runner.py (container execution)
- [ ] node/worker.py (main worker loop)
- [ ] node/models/qwen2.5-1.5b-instruct-q8_0.gguf (ALREADY EXISTS)

### Shared Files
- [ ] shared/__init__.py
- [ ] shared/models.py (Task, Result, TestResult dataclasses)
- [ ] shared/hivemind_utils.py (DHT helper functions)
- [ ] shared/prompts.py (LLM prompt templates)

### Docker Files
- [ ] docker/Dockerfile.sandbox (Python execution environment)

### Scripts
- [ ] scripts/setup_bootstrap.sh (initialize bootstrap node)
- [ ] scripts/start_worker.sh (worker startup with args)

### Configuration
- [ ] requirements.txt (all dependencies)
- [ ] .env.example (template for environment)
- [ ] .gitignore (ignore venv, models, cache)
- [ ] README.md (setup and usage instructions)

### Tests
- [ ] tests/test_orchestrator.py (unit tests)
- [ ] tests/test_node.py (unit tests for worker nodes)
- [ ] tests/test_integration.py (end-to-end tests)

---

## FUTURE V2 ENHANCEMENTS (NOT IN MVP)

The following features are planned for V2 but NOT part of MVP:

1. Voting mechanism (3 workers solve same task, majority wins)
2. RAG system (vector database for code snippets)
3. Web search integration (fetch documentation)
4. Evolution/fitness scores
5. Credit system for compute sharing
6. Task prioritization based on urgency
7. Multi-turn conversation for clarification
8. Support for languages beyond Python
9. Real distributed git (workers commit directly)
10. Advanced merge conflict resolution

---

## RISK MITIGATION

### Known Issues and Solutions

**Issue**: Qwen generates incorrect code
**Mitigation**: Docker testing catches errors, return failure status, retry with clarified prompt

**Issue**: DHT network partition
**Mitigation**: Implement timeout, fallback to centralized queue if DHT unavailable

**Issue**: Worker node crashes mid-task
**Mitigation**: Task expiration in DHT (300s), orchestrator retries uncompleted tasks

**Issue**: Token limit exceeded
**Mitigation**: Implement prompt truncation strategy, summarize p.o.a if needed

**Issue**: Git merge conflicts
**Mitigation**: For MVP, last write wins. V2 will implement proper conflict resolution

**Issue**: Docker container hangs
**Mitigation**: 30 second timeout, force kill container, mark task as failed

---

## NEXT STEPS FOR CLAUDE CODE

1. Create project structure with all directories (node/models already exists with model)
2. Implement shared/models.py (foundation for all other files)
3. Implement orchestrator/config.py and node/config.py
4. Implement shared/prompts.py with all 6 query templates
5. Implement orchestrator/task_decomposer.py (Query 5.1)
6. Implement node/task_executor.py (Queries 5.2, 5.3, 5.6)
7. Implement orchestrator/task_distributor.py and result_collector.py
8. Implement node/hivemind_client.py
9. Implement node/docker_runner.py
10. Implement node/worker.py (main loop)
11. Implement orchestrator/git_manager.py
12. Implement orchestrator/main.py (CLI)
13. Create docker/Dockerfile.sandbox
14. Write requirements.txt
15. Create .env.example and .gitignore
16. Write scripts for setup
17. Test locally with 1 orchestrator + 3 workers
18. Document and iterate

This plan provides complete specification for implementation. Each file has clear inputs, outputs, and processing logic. All queries from the PDF are incorporated with proper token limit handling.

**Repository already contains**: node/models/qwen2.5-1.5b-instruct-q8_0.gguf (1.1GB model file)
