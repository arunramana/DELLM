import json
import logging
import anthropic
from orchestrator.config import CLAUDE_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS
from shared.models import Task
from shared.prompts import TASK_DECOMPOSITION

logger = logging.getLogger(__name__)


def decompose_task(user_query: str) -> list[Task]:
    """Use Claude API to break a user query into subtasks."""
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    prompt = TASK_DECOMPOSITION.format(user_query=user_query)

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Extract JSON array from response
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON array found in response: {text}")

    tasks_data = json.loads(text[start:end])
    tasks = [Task.from_dict(t) for t in tasks_data]
    logger.info(f"Decomposed into {len(tasks)} tasks")
    return tasks
