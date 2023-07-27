from abc import ABC, abstractmethod

import numpy as np


class VisualizationDataContainer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def feature_count(self) -> int:
        """
        Return the number of features contained in the input sequence

        Returns
        -------
        int
            Number of features
        """
        pass

    @abstractmethod
    def length(self) -> int:
        """
        Return the number of elements of the input sequence. This value is equal across all features.

        Returns
        -------
        int
            Number of elements
        """
        pass

    @abstractmethod
    def input_sequence(self, feature: int = None) -> np.ndarray:
        """
        Return the input sequence of the specified feature as numpy array of shape (time,), if feature is not None.
        Otherwise, return the sequence with all features, i.e., of shape (time,feature)

        Parameters
        ----------
        feature
            Feature index

        Returns
        -------
        np.ndarray
            Input sequence
        """
        pass

    @abstractmethod
    def probability(self, feature: int = None) -> np.ndarray:
        """
        Return the probability distribution of the input sequence as numpy array with shape (time,position).
        If None, return (time,feature,position).

        The positions range from 0 to 127.

        Parameters
        ----------
        feature
            Feature index

        Returns
        -------
        np.ndarray
            Probability distribution
        """
        pass

    @abstractmethod
    def information_content(self, feature: int = None) -> np.ndarray:
        """
        Return the information content values of the input sequence as numpy array with shape (time,).
        If None, return (time,feature,).

        Parameters
        ----------
        feature
            Feature index

        Returns
        -------
        np.ndarray
            Information content values
        """
        pass

    @abstractmethod
    def entropy(self, feature: int = None) -> np.ndarray:
        """
        Return the entropy values of the input sequence as numpy array with shape (time,).
        If None, return (time,feature,).

        Parameters
        ----------
        feature
            Feature index

        Returns
        -------
        np.ndarray
            Information content values
        """
        pass
