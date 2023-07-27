from pathlib import Path
from typing import Union, Optional, List

from .base import Dataset, Composition, Viewpoint


class IdyomDatabase:
    def __init__(self):
        pass

    def import_midi_dataset(self, path: str) -> int:
        """
        Call IDyOM's import function of MIDI files using the specified path.

        Parameters
        ----------
        path
            Path must point to a directory. Directory must contain only MIDI files.

        Returns
        -------
        Id of the imported dataset.
        """
        pass

    def import_kern_dataset(self, path: str) -> int:
        """
        Call IDyOM's import function of **kern files using the specified path.

        Parameters
        ----------
        path
            Path must point to a directory. Directory must contain only **kern files.

        Returns
        -------
        Id of the imported dataset.
        """
        pass

    def get_all_datasets(self) -> list:
        """
        Return a list of all datasets available in IDyOM's database.

        Returns
        -------
        List of datasets
        """
        pass

    def get_dataset_alphabet(self, dataset: Union[int, Dataset]) -> list:
        """
        Return the alphabet of the dataset.

        Parameters
        ----------
        dataset
            Id of the dataset

        Returns
        -------
        List of all ever used symbols in the dataset
        """
        pass

    def get_all_compositions(self, dataset: Union[int, Dataset]) -> list:
        """
        Return all compositions within the specified dataset.

        Parameters
        ----------
        dataset
            Id or dataset object

        Returns
        -------
        List of all contained compositions
        """
        pass

    def encode_composition(self, composition: Union[int, Composition], viewpoint_spec: List[Viewpoint],
                           dataset: Union[int, Dataset] = None) -> list:
        """
        Transform a composition into a (or multiple) viewpoint sequence(s).

        Parameters
        ----------
        composition
            Composition object or composition id. If id, then dataset must not be None.
        viewpoint_spec
            List of viewpoints to transform the composition to.
        dataset
            Object or dataset id. Must be not None, if composition is specified by id.

        Returns
        -------
        A list of transformations of the specified composition.
        """
        pass
