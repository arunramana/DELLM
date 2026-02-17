import logging
from shared.models import Task
from shared.hivemind_utils import create_dht, store_value

logger = logging.getLogger(__name__)


def distribute_tasks(tasks: list[Task], initial_peers=None, dht_port=None):
    """Publish tasks to the Hivemind DHT for workers to pick up."""
    dht = create_dht(initial_peers=initial_peers, port=dht_port)
    logger.info(f"Connected to DHT, distributing {len(tasks)} tasks")

    # Store total task count so workers know how many to look for
    store_value(dht, "task_count", {"count": len(tasks)}, expiration_time=600.0)

    for task in tasks:
        key = f"task:{task.id}"
        store_value(dht, key, task.to_dict(), expiration_time=300.0)
        logger.info(f"Published {key}: {task.description}")

    return dht
