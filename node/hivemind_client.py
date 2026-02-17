import logging
from shared.models import Task
from shared.hivemind_utils import create_dht, get_value, store_value

logger = logging.getLogger(__name__)


class HivemindClient:
    def __init__(self, initial_peers=None, port=None):
        self.dht = create_dht(initial_peers=initial_peers, port=port)
        logger.info("Worker connected to DHT")

    def get_task_count(self) -> int:
        """Get total number of tasks from the DHT."""
        data = get_value(self.dht, "task_count")
        if data:
            return data.get("count", 0)
        return 0

    def find_available_task(self, max_task_id: int = 20) -> Task | None:
        """Scan DHT for an unclaimed task."""
        count = self.get_task_count()
        scan_range = max(count, max_task_id)

        for task_id in range(1, scan_range + 1):
            # Check if already claimed
            claimed = get_value(self.dht, f"status:claimed:{task_id}")
            if claimed:
                continue
            # Check if result already exists
            result = get_value(self.dht, f"result:{task_id}")
            if result:
                continue
            # Check if task exists
            task_data = get_value(self.dht, f"task:{task_id}")
            if task_data:
                return Task.from_dict(task_data)

        return None

    def claim_task(self, task_id: int, worker_id: str) -> bool:
        """Claim a task by writing status to DHT."""
        return store_value(
            self.dht,
            f"status:claimed:{task_id}",
            {"worker_id": worker_id},
            expiration_time=300.0,
        )

    def upload_result(self, result_data: dict, task_id: int) -> bool:
        """Upload a result to the DHT."""
        return store_value(
            self.dht,
            f"result:{task_id}",
            result_data,
            expiration_time=600.0,
        )
