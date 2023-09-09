from pathlib import Path
from typing import Union, List

import cl4py
from cl4py import Lisp
from .base import Dataset, Composition, Viewpoint, BasicViewpoint, transform_viewpoints_list_to_string_list
from .util import cl4py_cons_to_list
from ..lib.util import path_as_string_with_trailing_slash
import re


class IdyomDatabase:
    lisp: Lisp

    def __init__(self, idyom_root_path: Union[str, Path], idyom_sqlite_database_path: Union[str, Path]):
        """
        Class to execute commands within IDyOM

        Parameters
        ----------
        idyom_root_path
            Path to IDyOM's root (data) directory
        idyom_sqlite_database_path
            Path to IDyOM's sqlite database file
        """
        self.idyom_root_path: str = path_as_string_with_trailing_slash(idyom_root_path)
        self.idyom_sqlite_database_path: str = str(idyom_sqlite_database_path)

        self.lisp = cl4py.Lisp(quicklisp=True)
        self._setup_lisp()

    def _setup_lisp(self):
        self.eval(('defvar', 'common-lisp-user::*idyom-root*', '"' + self.idyom_root_path + '"'))
        self.eval(('with-open-file', ('*standard-output**', '"/dev/null"', ':direction', ':output',
                                      ':if-exists', ':supersede'),
                   ('load', ('SB-IMPL::USERINIT-PATHNAME',))
                   ))
        self.eval(('with-open-file', ('*standard-output**', '"/dev/null"', ':direction', ':output',
                                      ':if-exists', ':supersede'),
                   ('ql:quickload', '"clsql"'),
                   ('ql:quickload', '"idyom"'),
                   ('clsql:connect', ('list', '"' + self.idyom_sqlite_database_path + '"'), ':if-exists', ':old',
                    ':database-type', ':sqlite3')
                   ))

    def eval(self, expr) -> tuple:
        """
        Evaluate an expression in lisp.

        Parameters
        ----------
        expr
            Expression

        Returns
        -------
        tuple
            (return object, console output)
        """
        return self.lisp.eval(expr, True, False)

    def next_free_dataset_id(self) -> int:
        """
        Return next free dataset id of IDyOM's database

        Returns
        -------
        int
            Dataset id
        """
        return int(self.eval(('idyom-db:get-next-free-id',))[0])

    def import_midi_dataset(self, path: Union[str, Path],
                            description: str, dataset_id: int = None, timebase: int = 96) -> int:
        """
        Call IDyOM's import function of MIDI files using the specified path.

        Parameters
        ----------
        path
            Path must point to a directory. Directory must contain only MIDI files.
        description
            Description string, which also gets stored in IDyOM's database.
        dataset_id
            Id, which this imported dataset should youse. If None, the next available value is used.
        timebase
            IDyOM stores all note events' durations as integer numbers, based on a multiple
            of the timebase value. For example, a crotchet (quarter note) might be 1*96.
            When importing the Monophonic Corpus of Complete Compositions (MCCC), this value must be 39473280.

        Returns
        -------
        Id of the imported dataset.
        """
        if dataset_id is None:
            dataset_id = self.next_free_dataset_id()

        _, console_output = self.eval(
            ("let", (("kern2db::*default-timebase*", timebase),),
             ('idyom-db:import-data', ':mid', '"' + path_as_string_with_trailing_slash(path) + '"',
              '"' + description + '"', dataset_id))
        )
        result_dataset_id = int(re.findall(r"Inserting data into database: dataset (\d+)", console_output)[0])

        if dataset_id != result_dataset_id:
            raise RuntimeError("Unexpectedly, dataset_id does not match result_dataset_id!")

        return result_dataset_id

    def import_kern_dataset(self, path: Union[str, Path],
                            description: str, dataset_id: int = None, timebase: int = 96) -> int:
        """
        Call IDyOM's import function of **kern files using the specified path.

        Parameters
        ----------
        path
            Path must point to a directory. Directory must contain only **kern files.
        description
            Description string, which also gets stored in IDyOM's database.
        dataset_id
            Id, which this imported dataset should youse. If None, the next available value is used.
        timebase
            IDyOM stores all note events' durations as integer numbers, based on a multiple
            of the timebase value. For example, a crotchet (quarter note) might be 1*96.
            When importing the Monophonic Corpus of Complete Compositions (MCCC), this value must be 39473280.

        Returns
        -------
        Id of the imported dataset.
        """
        if dataset_id is None:
            dataset_id = self.next_free_dataset_id()

        _, console_output = self.eval(
            ("let", (("kern2db::*default-timebase*", timebase),),
             ('idyom-db:import-data', ':krn', '"' + path_as_string_with_trailing_slash(path) + '"',
              '"' + description + '"', dataset_id))
        )
        result_dataset_id = int(re.findall(r"Inserting data into database: dataset (\d+)", console_output)[0])

        if dataset_id != result_dataset_id:
            raise RuntimeError("Unexpectedly, dataset_id does not match result_dataset_id!")

        return result_dataset_id

    def get_all_datasets(self) -> list:
        """
        Return a list of all datasets available in IDyOM's database.

        Returns
        -------
        list
            List of datasets
        """
        result = list()

        try:
            (_, console_output) = self.eval(('idyom-db:describe-database',))

            for line in console_output.split("\n"):
                id, description = re.split(r"\s+", line, maxsplit=1)
                result.append(Dataset(id=int(id), description=description.strip()))
        except:  # If there is no dataset in the database (or any other error)
            pass
        return result

    def get_dataset_alphabet(self, datasets: Union[Union[int, Dataset], List[Union[int, Dataset]]],
                             viewpoint: BasicViewpoint) -> list:
        """
        Return the shared alphabet of the datasets.

        Parameters
        ----------
        datasets
            Ids of the datasets
        viewpoint
            Viewpoint to use when determining the alphabet

        Returns
        -------
        list
            List of all ever used symbols (encoded as the specified viewpoint) in the datasets
        """
        if isinstance(datasets, int):
            datasets = [datasets]
        elif isinstance(datasets, Dataset):
            datasets = [datasets.id]
        elif isinstance(datasets, list):
            _tmp = []
            for e in datasets:
                if isinstance(e, int):
                    _tmp.append(e)
                elif isinstance(e, Dataset):
                    _tmp.append(e.id)
                else:
                    raise ValueError("datasets invalid! If list, each element must be either of type int or Dataset.")
            datasets = _tmp
        else:
            raise ValueError("datasets invalid! Value must either be of type int, Dataset, or a list of these types.")

        result, _ = self.eval(('idyom-db::get-alphabet', ('quote', viewpoint.value), *datasets))
        alphabet = cl4py_cons_to_list(result)

        return alphabet

    def get_all_compositions(self, dataset: Union[int, Dataset]) -> List[Composition]:
        """
        Return all compositions within the specified dataset.

        Parameters
        ----------
        dataset
            Id or dataset object

        Returns
        -------
        List[Composition]
            List of all contained compositions
        """
        if isinstance(dataset, Dataset):
            dataset = dataset.id

        descriptions = self.eval(
            ("mapcar", ("function", "idyom-db::composition-description"), ("idyom-db:get-compositions", dataset)))[0]
        composition_ids = self.eval((
            "mapcan", ("function", ("lambda", ("x",), ("cdr", ("idyom-db:get-id", "x")))),
            ("idyom-db:get-compositions", dataset)))[0]
        result = list()
        for idx, description in enumerate(
                descriptions):  # assuming that idx always coincides with the composition's id within the dataset
            result.append(Composition(dataset_id=dataset, id=composition_ids[idx], description=description))

        return result

    def encode_composition(self, composition: Union[int, Composition],
                           viewpoint_spec: Union[Viewpoint, List[Viewpoint]],
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
            Object or dataset id. Must not be None, if composition is specified by id.

        Returns
        -------
        A list of transformations of the specified composition.
        """
        if isinstance(composition, Composition):
            dataset = composition.dataset_id
            composition = composition.id # intentionally set composition to id
        else:
            if dataset is not None:
                if isinstance(dataset, Dataset):
                    dataset = dataset.id
            else:
                raise ValueError("dataset invalid! If composition is not of type Composition, dataset must not be None.")

        cmd = ("viewpoints:viewpoint-sequence", (
            "viewpoints:get-viewpoint", ("quote", tuple(transform_viewpoints_list_to_string_list(viewpoint_spec)))),
               ("md:get-event-sequence", dataset, composition))
        try:
            viewpoint_sequence, _ = self.eval(cmd)
            viewpoint_sequence = cl4py_cons_to_list(viewpoint_sequence)
        except:
            print("Error while executing", cmd)
            viewpoint_sequence = []

        return viewpoint_sequence

