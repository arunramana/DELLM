TASK_DECOMPOSITION = """Given {user_query}:
I need you to understand the user query and break it down into subproblems or subtasks that can be solved by a network of SLMs to solve the user's query.

Create a concise and clear plan of action to break down the user's requirement into smaller tasks. The task should include:
- inputs and outputs required
- directory and repository to the file where the code has to be written to
- context of the task it has to solve
- Test each piece of code generated in each node

Ensure the tasks are completed in the correct order in p.o.a.
The p.o.a should have the list of all the tasks.

Return as valid JSON array with this structure:
[{{
  "id": 1,
  "description": "Create models.py with User table",
  "file": "models.py",
  "dependencies": [],
  "input_output_spec": "Input: None, Output: SQLAlchemy User model class",
  "directory": "app/",
  "context": "Database models for user authentication"
}}]

Return ONLY the JSON array, no other text."""

COMPLEXITY_CHECK = """Given:
- User query: {user_query}
- Plan of action: {poa}
- Task: {task}
- Input/Output: {input_output}
- Directory: {directory}

Can the given task be completely solved by this SLM?
Answer ONLY 'yes' or 'no'."""

TASK_DIVISION = """Given:
- User query: {user_query}
- Plan of action: {poa}
- Task: {task}
- Input/Output: {input_output}
- Directory: {directory}

Divide the task into 2-4 subtasks so they can be solved by other nodes.
Try to minimize the number of subtasks needed to solve this task.

Return as JSON array:
[{{
  "subtask_description": "...",
  "subtask_input_output": "...",
  "subtask_order": 1
}}]

Return ONLY the JSON array, no other text."""

CODE_GENERATION = """Given:
- User query: {user_query}
- Plan of action: {poa}
- Task: {task}
- Input/Output: {input_output}
- Directory: {directory}

Solve the task:
1. Ensure inputs and outputs match properly
2. Write test cases to test this code
3. Write comments to document and reason
4. Generate production-ready Python code

Output ONLY the code, no explanations, no markdown."""

FINAL_SUMMARY = """Given:
- User query: {user_query}
- Plan of action: {poa}

Now that all the code has been generated, tell the user what has been done to solve their query and how to run and test the code. Also mention the repository where the code was modified.

Keep response under 200 words."""
