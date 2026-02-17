import logging
from llama_cpp import Llama
from shared.models import Task
from shared.prompts import COMPLEXITY_CHECK, TASK_DIVISION, CODE_GENERATION
from node.config import MODEL_PATH, MODEL_N_CTX, MODEL_N_THREADS, MODEL_TEMPERATURE, MODEL_MAX_TOKENS

logger = logging.getLogger(__name__)

_llm = None


def get_llm() -> Llama:
    """Lazy-load the LLM model."""
    global _llm
    if _llm is None:
        logger.info(f"Loading model from {MODEL_PATH}...")
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=MODEL_N_CTX,
            n_threads=MODEL_N_THREADS,
            verbose=False,
        )
        logger.info("Model loaded")
    return _llm


def _query_llm(prompt: str) -> str:
    """Send a prompt to the local Qwen model and return the response."""
    llm = get_llm()
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=MODEL_TEMPERATURE,
        max_tokens=MODEL_MAX_TOKENS,
    )
    return response["choices"][0]["message"]["content"].strip()


def check_complexity(task: Task, user_query: str, poa: str) -> bool:
    """Check if the SLM can handle this task (Query 5.2)."""
    prompt = COMPLEXITY_CHECK.format(
        user_query=user_query,
        poa=poa,
        task=task.description,
        input_output=task.input_output_spec,
        directory=task.directory,
    )
    answer = _query_llm(prompt).lower().strip()
    return "yes" in answer


def generate_code(task: Task, user_query: str, poa: str) -> str:
    """Generate code for a task (Query 5.6)."""
    prompt = CODE_GENERATION.format(
        user_query=user_query,
        poa=poa,
        task=task.description,
        input_output=task.input_output_spec,
        directory=task.directory,
    )
    code = _query_llm(prompt)
    # Strip markdown fences if present
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:]  # Remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code


def execute_task(task: Task, user_query: str, poa: str) -> str:
    """Execute a full task: check complexity, then generate code."""
    can_handle = check_complexity(task, user_query, poa)
    if not can_handle:
        logger.info(f"Task {task.id} flagged as complex, attempting anyway")
    code = generate_code(task, user_query, poa)
    logger.info(f"Generated {len(code)} chars of code for task {task.id}")
    return code
