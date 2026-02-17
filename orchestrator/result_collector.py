import time
import logging
from shared.models import Task, Result
from shared.hivemind_utils import get_value
from orchestrator.config import TASK_TIMEOUT, RESULT_COLLECTION_POLL_INTERVAL

logger = logging.getLogger(__name__)


def collect_results(dht, tasks: list[Task], timeout: int = TASK_TIMEOUT) -> dict[int, Result]:
    """Poll DHT for task results until all collected or timeout."""
    results = {}
    task_ids = {t.id for t in tasks}
    start_time = time.time()

    while len(results) < len(tasks):
        elapsed = time.time() - start_time
        if elapsed > timeout:
            missing = task_ids - set(results.keys())
            logger.warning(f"Timeout after {timeout}s. Missing tasks: {missing}")
            break

        for task in tasks:
            if task.id in results:
                continue
            key = f"result:{task.id}"
            data = get_value(dht, key)
            if data:
                results[task.id] = Result.from_dict(data)
                logger.info(f"Collected result for task {task.id}")

        if len(results) < len(tasks):
            time.sleep(RESULT_COLLECTION_POLL_INTERVAL)

    return results
