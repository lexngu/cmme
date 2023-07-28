from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from cmme.lib.instructions_file import InstructionsFile
from cmme.lib.results_file import ResultsFile


class ModelBuilder(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def to_instructions_file(self) -> InstructionsFile:
        raise NotImplementedError


class Model(ABC):
    def __init__(self):
        pass

    @classmethod
    def run_instructions_file(cls, instructions_file: InstructionsFile) -> ResultsFile:
        with tempfile.NamedTemporaryFile() as tmpfile:
            instructions_file.save_self(tmpfile.name)
            return cls.run_instructions_file_at_path(tmpfile.name)

    @staticmethod
    @abstractmethod
    def run_instructions_file_at_path(file_path) -> ResultsFile:
        raise NotImplementedError
