from __future__ import annotations

from abc import ABC, abstractmethod


class InstructionsFile(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def _generate_instructions_file_path() -> str:
        """
        Generate a default value of the file path where to store an instructions file.

        Returns
        -------
        File path
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _save(cls, instructions_file: InstructionsFile, file_path: str):
        """
        Write to disk.

        Parameters
        ----------
        instructions_file
            Object to write to disk
        file_path
            File path where to write to
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def save(cls, instructions_file: InstructionsFile, file_path: str):
        """
        Write the result file to disk.

        Parameters
        ----------
        instructions_file
            result file object to write to disk
        file_path
            File path where to write to
        """
        cls._save(instructions_file, file_path)

    def save_self(self, file_path: str):
        """
        Save this result file object as file.

        Parameters
        ----------
        file_path
            Where to store the result file
        """
        self.save(self, file_path)

    @staticmethod
    @abstractmethod
    def load(file_path: str) -> InstructionsFile:
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
