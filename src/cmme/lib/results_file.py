from __future__ import annotations

from abc import ABC, abstractmethod


class ResultsFile(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def _generate_results_file_path() -> str:
        """
        Generate a default value of the file path where to store a result file.

        Returns
        -------
        File path
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _save(results_file: ResultsFile, file_path: str):
        """
        Write to disk.

        Parameters
        ----------
        results_file
            Object to write to disk
        file_path
            File path where to write to
        """
        raise NotImplementedError

    @classmethod
    def save(cls, results_file: ResultsFile, file_path: str):
        """
        Write the result file to disk.

        Parameters
        ----------
        results_file
            result file object to write to disk
        file_path
            File path where to write to
        """
        return cls._save(results_file, file_path)

    def save_self(self, file_path: str):
        """
        Save this result file object as file.

        Parameters
        ----------
        file_path
            Where to store the result file
        """
        return self.save(self, file_path)

    @staticmethod
    @abstractmethod
    def load(file_path: str) -> ResultsFile:
        """
        Load a results file.

        Parameters
        ----------
        file_path
            Where the results file to load is stored

        Returns
        -------
        ResultsFile
            Loaded results file
        """
        raise NotImplementedError
