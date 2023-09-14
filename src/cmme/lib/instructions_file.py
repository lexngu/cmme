from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


class InstructionsFile(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def save(instructions_file: InstructionsFile, instructions_file_path: Union[str, Path],
             results_file_path: Union[str, Path] = None):
        """
        Write the instructions file to disk.

        Parameters
        ----------
        instructions_file
            instructions file object to write to disk
        instructions_file_path
            File path where to write to
        results_file_path
            File path where to write the results to. If None, the external intermediate scripts will provide a value
            by their own when processing the instructions file.
        """
        raise NotImplementedError

    def save_self(self, instructions_file_path: Union[str, Path], results_file_path: Union[str, Path] = None):
        """
        Save this result file object as file.

        Parameters
        ----------
        instructions_file_path
            Where to store the result file
        results_file_path
            File path where to write the results to. If None, the external intermediate scripts will provide a value
            by their own when processing the instructions file.
        """
        self.save(self, instructions_file_path, results_file_path)

    @staticmethod
    @abstractmethod
    def load(file_path: Union[str, Path]) -> InstructionsFile:
        """
        Load an instructions file.

        Parameters
        ----------
        file_path
            Where the instructions file to load is stored

        Returns
        -------
        ResultsFile
            Loaded instructions file
        """
        raise NotImplementedError
