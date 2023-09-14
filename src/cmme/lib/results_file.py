from __future__ import annotations

from abc import ABC, abstractmethod


class ResultsFile(ABC):
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def save(cls, results_file: ResultsFile, file_path: str):
        """
        Write the result file to disk.

        Parameters
        ----------
        results_file
            Result file object to write to disk
        file_path
            File path where to write to
        """
        return NotImplementedError

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
