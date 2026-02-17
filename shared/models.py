from dataclasses import dataclass, asdict
from typing import List
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
    def from_dict(data: dict) -> "Task":
        return Task(**data)

    @staticmethod
    def from_json(json_str: str) -> "Task":
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
    def from_dict(data: dict) -> "Result":
        return Result(**data)

    @staticmethod
    def from_json(json_str: str) -> "Result":
        return Result.from_dict(json.loads(json_str))
