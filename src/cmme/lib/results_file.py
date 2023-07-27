from __future__ import annotations

from abc import ABC, abstractmethod


class ResultsFile(ABC):
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def _generate_results_file_path(cls) -> str:
        """
        Generate a default value of the file path where to store an result file.

        Returns
        -------
        File path
        """
        pass

    @classmethod
    @abstractmethod
    def _save(cls, results_file: ResultsFile, file_path):
        """
        Write to disk.

        Parameters
        ----------
        results_file
            Object to write to disk
        file_path
            File path where to write to
        """
        pass

    @classmethod
    def save(cls, results_file: ResultsFile, file_path=_generate_results_file_path()):
        """
        Write the result file to disk.

        Parameters
        ----------
        results_file
            result file object to write to disk
        file_path
            File path where to write to
        """
        return ResultsFile._save(results_file, file_path)

    def save_self(self, file_path: str):
        """
        Save this result file object to the disk.

        Parameters
        ----------
        file_path
            Where to store the result file
        """
        return ResultsFile.save(self, file_path)

    @classmethod
    @abstractmethod
    def load(cls, file_path) -> ResultsFile:
        pass
