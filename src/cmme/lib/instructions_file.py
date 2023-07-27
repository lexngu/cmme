from __future__ import annotations

from abc import ABC, abstractmethod


class InstructionsFile(ABC):
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def _generate_instructions_file_path(cls) -> str:
        """
        Generate a default value of the file path where to store an instruction file.

        Returns
        -------
        File path
        """
        pass

    @classmethod
    @abstractmethod
    def _save(cls, instructions_file: InstructionsFile, file_path):
        """
        Write to disk.

        Parameters
        ----------
        instructions_file
            Object to write to disk
        file_path
            File path where to write to
        """
        pass

    @classmethod
    def save(cls, instructions_file: InstructionsFile, file_path=_generate_instructions_file_path()):
        """
        Write the instruction file to disk.

        Parameters
        ----------
        instructions_file
            Instruction file object to write to disk
        file_path
            File path where to write to
        """
        return InstructionsFile._save(instructions_file, file_path)

    def save_self(self, file_path: str):
        """
        Save this instruction file object to the disk.

        Parameters
        ----------
        file_path
            Where to store the instruction file
        """
        return InstructionsFile.save(self, file_path)

    @classmethod
    @abstractmethod
    def load(cls, file_path) -> InstructionsFile:
        pass
