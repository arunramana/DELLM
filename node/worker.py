import time
import logging
import json

from node.config import BOOTSTRAP_PEERS, DHT_PORT, WORKER_ID, TASK_POLL_INTERVAL
from node.hivemind_client import HivemindClient
from node.task_executor import execute_task
from node.docker_runner import run_code_in_docker
from shared.models import Result
from shared.hivemind_utils import get_value

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_poa_and_query(client: HivemindClient) -> tuple[str, str]:
    """Try to reconstruct user query and plan of action from available tasks."""
    tasks = []
    count = client.get_task_count()
    for task_id in range(1, count + 1):
        data = get_value(client.dht, f"task:{task_id}")
        if data:
            tasks.append(data)

    if not tasks:
        return "", ""

    # Use first task's context as user query approximation
    user_query = tasks[0].get("context", "")
    poa = json.dumps([{"id": t["id"], "description": t["description"], "file": t["file"]} for t in tasks])
    return user_query, poa


def worker_loop():
    """Main worker loop: poll for tasks, execute, upload results."""
    logger.info(f"Starting worker {WORKER_ID}")
    client = HivemindClient(initial_peers=BOOTSTRAP_PEERS, port=DHT_PORT)
    logger.info("Worker connected to DHT, polling for tasks...")

    while True:
        task = client.find_available_task()
        if task is None:
            time.sleep(TASK_POLL_INTERVAL)
            continue

        logger.info(f"Found task {task.id}: {task.description}")

        # Claim it
        if not client.claim_task(task.id, WORKER_ID):
            logger.warning(f"Failed to claim task {task.id}, skipping")
            continue

        logger.info(f"Claimed task {task.id}")

        # Get context
        user_query, poa = get_poa_and_query(client)

        # Generate code
        try:
            code = execute_task(task, user_query, poa)
        except Exception as e:
            logger.error(f"Code generation failed for task {task.id}: {e}")
            code = f"# Code generation failed: {e}"

        # Test in Docker
        try:
            test_result = run_code_in_docker(code, filename=task.file)
        except Exception as e:
            logger.error(f"Docker test failed for task {task.id}: {e}")
            from shared.models import TestResult
            test_result = TestResult(
                success=False, stdout="", stderr=str(e),
                tests_passed=0, tests_failed=1, execution_time=0.0,
            )

        # Upload result
        result = Result(
            task_id=task.id,
            code=code,
            test_passed=test_result.success,
            test_logs=test_result.stdout + "\n" + test_result.stderr,
            execution_time=test_result.execution_time,
            worker_id=WORKER_ID,
            timestamp=time.time(),
        )

        client.upload_result(result.to_dict(), task.id)
        logger.info(f"Uploaded result for task {task.id} (passed={test_result.success})")


if __name__ == "__main__":
    worker_loop()
