from __future__ import annotations

from abc import ABC, abstractmethod
from cmme.lib.instructions_file import InstructionsFile
from cmme.lib.results_file import ResultsFile


class ModelBuilder(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def to_instructions_file(self) -> InstructionsFile:
        pass


class Model(ABC):
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def run_instructions_file(instructions_file: InstructionsFile) -> ResultsFile:
        pass

    @classmethod
    @abstractmethod
    def run_instructions_file_at_path(file_path) -> ResultsFile:
        pass