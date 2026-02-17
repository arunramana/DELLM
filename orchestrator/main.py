import argparse
import logging
import sys

from orchestrator.config import BOOTSTRAP_PEERS, DHT_PORT
from orchestrator.task_decomposer import decompose_task
from orchestrator.task_distributor import distribute_tasks
from orchestrator.result_collector import collect_results
from orchestrator.git_manager import merge_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DELLM Orchestrator")
    parser.add_argument("--query", required=True, help="Coding task description")
    parser.add_argument("--repo", required=True, help="Target git repository path")
    parser.add_argument("--timeout", type=int, default=120, help="Result collection timeout in seconds")
    args = parser.parse_args()

    logger.info(f"Query: {args.query}")

    # Step 1: Decompose task
    logger.info("Decomposing task via Claude API...")
    tasks = decompose_task(args.query)
    for t in tasks:
        logger.info(f"  Task {t.id}: {t.description} -> {t.directory}{t.file}")

    # Step 2: Distribute tasks via DHT
    logger.info("Distributing tasks to DHT...")
    dht = distribute_tasks(tasks, initial_peers=BOOTSTRAP_PEERS, dht_port=DHT_PORT)

    # Step 3: Collect results
    logger.info("Waiting for worker results...")
    results = collect_results(dht, tasks, timeout=args.timeout)

    # Step 4: Merge to git
    logger.info("Merging results to git repo...")
    commit_hash = merge_results(tasks, results, args.repo)

    # Step 5: Summary
    total = len(tasks)
    collected = len(results)
    passed = sum(1 for r in results.values() if r.test_passed)
    failed = collected - passed

    print(f"\nTask completed!\n")
    print(f"Generated {collected}/{total} files:")
    for task in tasks:
        if task.id in results:
            r = results[task.id]
            status = "passed tests" if r.test_passed else "FAILED tests"
            print(f"  - {task.directory}{task.file} ({status})")
        else:
            print(f"  - {task.directory}{task.file} (NOT received)")

    print(f"\nLocation: {args.repo}")
    if commit_hash:
        print(f"Commit: {commit_hash}")
    print(f"\nTest Results Summary:")
    print(f"  - {passed} tasks passed tests")
    print(f"  - {failed} tasks failed tests")
    print(f"  - {total - collected} tasks not received")


if __name__ == "__main__":
    main()
