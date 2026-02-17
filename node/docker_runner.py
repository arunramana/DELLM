import os
import time
import tempfile
import logging
import docker
from shared.models import TestResult
from node.config import DOCKER_IMAGE, DOCKER_TIMEOUT

logger = logging.getLogger(__name__)


def run_code_in_docker(code: str, filename: str = "main.py") -> TestResult:
    """Run generated code in an isolated Docker container."""
    start_time = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write code file
        code_path = os.path.join(tmpdir, filename)
        with open(code_path, "w") as f:
            f.write(code)

        # Write a simple syntax/import test
        test_path = os.path.join(tmpdir, "test_syntax.py")
        with open(test_path, "w") as f:
            f.write(f"""import ast
import sys

# Test 1: syntax check
try:
    with open("/code/{filename}") as f:
        source = f.read()
    ast.parse(source)
    print("SYNTAX_OK")
except SyntaxError as e:
    print(f"SYNTAX_ERROR: {{e}}")
    sys.exit(1)

# Test 2: import check
try:
    sys.path.insert(0, "/code")
    exec(compile(source, "{filename}", "exec"))
    print("EXEC_OK")
except Exception as e:
    print(f"EXEC_ERROR: {{e}}")
    sys.exit(1)
""")

        try:
            client = docker.from_env()
            container = client.containers.run(
                DOCKER_IMAGE,
                command=f"timeout {DOCKER_TIMEOUT}s python /code/test_syntax.py",
                volumes={tmpdir: {"bind": "/code", "mode": "ro"}},
                mem_limit="512m",
                cpuset_cpus="0",
                network_mode="none",
                cap_drop=["ALL"],
                user="1000:1000",
                remove=True,
                detach=False,
                stdout=True,
                stderr=True,
            )

            output = container.decode("utf-8") if isinstance(container, bytes) else str(container)
            elapsed = time.time() - start_time
            success = "SYNTAX_OK" in output

            return TestResult(
                success=success,
                stdout=output,
                stderr="",
                tests_passed=1 if success else 0,
                tests_failed=0 if success else 1,
                execution_time=elapsed,
            )

        except docker.errors.ContainerError as e:
            elapsed = time.time() - start_time
            return TestResult(
                success=False,
                stdout=e.stderr.decode("utf-8") if e.stderr else "",
                stderr=str(e),
                tests_passed=0,
                tests_failed=1,
                execution_time=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Docker error: {e}")
            return TestResult(
                success=False,
                stdout="",
                stderr=str(e),
                tests_passed=0,
                tests_failed=1,
                execution_time=elapsed,
            )
