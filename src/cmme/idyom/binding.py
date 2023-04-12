import os
from typing import List, re, Any, Tuple

import cl4py

from .base import *
from .util import path_with_trailing_slash


class IDYOMBinding:
    def __init__(self, idyom_root_path, idyom_sqlite_database_path):
        self.lisp = cl4py.Lisp(quicklisp=True)
        self.idyom_root_path = path_with_trailing_slash(idyom_root_path)
        self.idyom_sqlite_database_path: Path = path_with_trailing_slash(idyom_sqlite_database_path)

        self._setup_lisp()

    def _lisp_eval(self, cmd):
        print("> " + str(cmd))
        result = self.lisp.eval(cmd)
        #print(str(result))
        return result

    def _setup_lisp(self):
        self._lisp_eval( ('defvar', 'common-lisp-user::*idyom-root*', '"'+self.idyom_root_path+'"') )
        self._lisp_eval( ('ql:quickload', '"idyom"') )
        self._lisp_eval( ('clsql:connect', ('list', '"'+self.idyom_sqlite_database_path+'"'), ':if-exists', ':old',
                   ':database-type', ':sqlite3') )

    def all_datasets(self) -> List[Dataset]:
        """
        Calls (idyom-db:describe-database)
        :return: A list of Dataset objects
        """
        result = list()

        self._lisp_eval( ('idyom-db:describe-database', ) )
        last_msg = self.lisp.msg # Requires the patched version of cl4py

        for line in last_msg.split("\n"):
            id, description = re.split("\s+", line, maxsplit=1)
            result.append(Dataset(id=int(id), description=description.strip()))

        return result

    def next_free_dataset_id(self) -> int:
        result = self._lisp_eval( ('idyom-db:get-next-free-id',) )
        return int(result)

    def import_midi(self, midi_files_directory_path: str, description: str, dataset_id: int) -> Dataset:
        """
        Calls (idyom-db:import-data :mid <midi_file_directory_path> <description> <dataset_id>)

        :param midi_files_directory_path: Path to directory containing midi files to import
        :param description: A string as description of the dataset
        :param dataset_id: Target dataset id. Note that if there already is a dataset with the provided id, IDyOM will fail.
        :return:
        """
        self._lisp_eval(('idyom-db:import-data', ':mid', '"' + path_with_trailing_slash(midi_files_directory_path) + '"', '"'+description+'"', dataset_id))
        dataset_id = int(re.findall("Inserting data into database: dataset (\d+)", self.lisp.msg)[0])

        return Dataset(id=dataset_id, description=description)

    def import_kern(self, krn_files_directory_path: str, description: str, dataset_id: int) -> Dataset:
        """
        Calls (idyom-db:import-data :mid <krn_files_directory_path> <description> <dataset_id>)

        :param krn_files_directory_path: Path to directory containing **kern files to import
        :param description: A string as description of the dataset
        :param dataset_id: Target dataset id. Note that if there already is a dataset with the provided id, IDyOM will fail.
        :return:
        """
        self._lisp_eval(('idyom-db:import-data', ':krn', '"'+path_with_trailing_slash(krn_files_directory_path)+'"', '"'+description+'"', dataset_id))
        dataset_id = int(re.findall("Inserting data into database: dataset (\d+)", self.lisp.msg)[0])

        return Dataset(id=dataset_id, description=description)

    def eval(self, cl4py_instruction: Tuple) -> Any:
        result = self._lisp_eval( cl4py_instruction )

        return result

    def all_compositions(self, dataset: Dataset) -> List[Composition]:
        descriptions = self._lisp_eval( ("mapcar", ("function", "idyom-db::composition-description"), ("idyom-db:get-compositions", dataset.id)) )
        result = list()
        for idx, description in enumerate(descriptions): # assuming that idx always coincides with the composition's id within the dataset
            result.append(Composition(dataset_id=dataset.id, id=idx, description=description))

        return result

    def derive_viewpoint_sequence(self, composition: Composition, viewpoints: List[Viewpoint]) -> List:
        viewpoint_sequence = self._lisp_eval( ("viewpoints:viewpoint-sequence", ("viewpoints:get-viewpoint", ("quote", viewpoints_list_as_lisp_string(viewpoints))), ("md:get-event-sequence", composition.dataset_id, composition.id)) )
        viewpoint_sequence = list(map(lambda x: x, viewpoint_sequence)) # convert Cons to list
        return viewpoint_sequence

    def get_alphabet(self, datasets, viewpoint: BasicViewpoint) -> List:
        if isinstance(datasets, Dataset):
            dataset_ids = [str(datasets.id)]
        elif isinstance(datasets, list):
            dataset_ids = list(map(lambda o: str(o.id), datasets))

        alphabet = self._lisp_eval( ("idyom-db::get-alphabet", ("quote", viewpoint.value), " ".join(dataset_ids)) )
        alphabet = list(map(lambda x: x, alphabet))  # convert Cons to list
        return alphabet