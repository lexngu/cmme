from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from cmme.lib.instructions_file import InstructionsFile
from cmme.lib.io import new_filepath
from cmme.lib.results_file import ResultsFile


class ModelBuilder(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def to_instructions_file(self) -> InstructionsFile:
        """
        Return instruction file object representing the currently model builder configuration.

        Returns
        -------
        InstructionsFile
            Instruction file object
        """
        raise NotImplementedError


class Model(ABC):
    def __init__(self):
        pass

    @classmethod
    def run_instructions_file(cls, instructions_file: InstructionsFile) -> ResultsFile:
        match cls.__name__:
            case "IDYOMModel":
                extension = "lisp"
            case "PPMModel":
                extension = "feather"
            case "DREXModel":
                extension = "mat"
            case _:
                extension = None
        if_path = new_filepath(cls.__name__, extension)
        instructions_file.save_self(if_path)
        print("Instructions file written to {}".format(if_path))
        return cls.run_instructions_file_at_path(if_path)

    @staticmethod
    @abstractmethod
    def run_instructions_file_at_path(file_path) -> ResultsFile:
        raise NotImplementedError
