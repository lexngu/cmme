from pathlib import Path
import pandas as pd
from typing import Any, Tuple, Union
import os
import cl4py
import re

from .base import *
from cmme.util import path_as_string_with_trailing_slash
from .util import cl4py_cons_to_list


def install_idyom(idyom_root_path: Union[str, Path], force_reset = False) -> Path:
    """
    Installs idyom as quicklisp project, and initializes the database
    :param idyom_root_path:
    :param force_reset:
    :return: Path to idyom database
    """
    idyom_repository_path = (Path(__file__) / Path("../../../../res/models/idyom/")).resolve()
    quicklisp_local_package_path = Path("~/quicklisp/local-projects/").expanduser().resolve()
    idyom_repository_symlink_target_path = (quicklisp_local_package_path / "idyom/")
    if not quicklisp_local_package_path.exists():
        quicklisp_local_package_path.mkdir(parents=True)
    if not idyom_repository_symlink_target_path.exists():
        idyom_repository_symlink_target_path.symlink_to(idyom_repository_path)

    lisp = cl4py.Lisp(quicklisp=True)

    idyom_root_path = path_as_string_with_trailing_slash(idyom_root_path)
    idyom_db_path = Path(os.path.join(idyom_root_path, "db/database.sqlite"))

    if idyom_db_path.exists() and not force_reset:
        print("Database at {} already exists. Use force_reset, if you want to reset the database.".format(idyom_db_path))

    if not idyom_db_path.parent.exists():
        Path(idyom_db_path).parent.mkdir(parents=True)

    lisp.eval(('defvar', 'common-lisp-user::*idyom-root*', '"' + idyom_root_path + '"'))
    lisp.eval(('ql:quickload', '"idyom"'))
    lisp.eval(('clsql:connect', (
    'list', '"' + str(idyom_db_path) + '"'), ':if-exists', ':old', ':database-type', ':sqlite3'))
    lisp.eval(('idyom-db:initialise-database',))

    return idyom_db_path


class IDYOMBinding:
    def __init__(self, idyom_root_path, idyom_sqlite_database_path):
        self.lisp = cl4py.Lisp(quicklisp=True)
        self.idyom_root_path: str = path_as_string_with_trailing_slash(idyom_root_path)
        self.idyom_sqlite_database_path: str = str(idyom_sqlite_database_path)

        self._setup_lisp()

    def _lisp_eval(self, cmd):
        #print("> " + str(cmd))
        result = self.lisp.eval(cmd)
        #print(str(result))
        return result

    def _setup_lisp(self):
        self._lisp_eval( ('defvar', 'common-lisp-user::*idyom-root*', '"'+ self.idyom_root_path +'"') )
        self._lisp_eval( ('ql:quickload', '"idyom"') )
        self._lisp_eval( ('clsql:connect', ('list', '"'+ self.idyom_sqlite_database_path +'"'), ':if-exists', ':old',
                   ':database-type', ':sqlite3') )

    def all_datasets(self) -> List[Dataset]:
        """
        Calls (idyom-db:describe-database)
        :return: A list of Dataset objects
        """
        result = list()

        try:
            self._lisp_eval( ('idyom-db:describe-database', ) )
            last_msg = self.lisp.msg # Requires the patched version of cl4py

            for line in last_msg.split("\n"):
                id, description = re.split("\s+", line, maxsplit=1)
                result.append(Dataset(id=int(id), description=description.strip()))
        except: # If there is no dataset in the database (or any other error)
            pass
        return result

    def next_free_dataset_id(self) -> int:
        result = self._lisp_eval( ('idyom-db:get-next-free-id',) )
        return int(result)

    def import_midi(self, midi_files_directory_path: str, description: str, dataset_id: int, timebase: int) -> Dataset:
        """
        Calls (idyom-db:import-data :mid <midi_file_directory_path> <description> <dataset_id>)

        :param midi_files_directory_path: Path to directory containing midi files to import
        :param description: A string as description of the dataset
        :param dataset_id: Target dataset id. Note that if there already is a dataset with the provided id, IDyOM will fail.
        :param timebase: Value of "kern2db::*default-timebase*" to use
        :return:
        """
        self._lisp_eval(
            ("let", (("kern2db::*default-timebase*", timebase),),
             ('idyom-db:import-data', ':mid', '"' + path_as_string_with_trailing_slash(midi_files_directory_path) + '"', '"'+description+'"', dataset_id))
        )
        dataset_id = int(re.findall("Inserting data into database: dataset (\d+)", self.lisp.msg)[0])

        return Dataset(id=dataset_id, description=description)

    def import_kern(self, krn_files_directory_path: str, description: str, dataset_id: int, timebase: int) -> Dataset:
        """
        Calls (idyom-db:import-data :mid <krn_files_directory_path> <description> <dataset_id>)

        :param krn_files_directory_path: Path to directory containing **kern files to import
        :param description: A string as description of the dataset
        :param dataset_id: Target dataset id. Note that if there already is a dataset with the provided id, IDyOM will fail.
        :param timebase: Value of "kern2db::*default-timebase*" to use
        :return:
        """
        self._lisp_eval(
            ("let", (("kern2db::*default-timebase*", timebase),),
             ('idyom-db:import-data', ':krn', '"'+ path_as_string_with_trailing_slash(krn_files_directory_path) +'"', '"'+description+'"', dataset_id))
        )
        dataset_id = int(re.findall("Inserting data into database: dataset (\d+)", self.lisp.msg)[0])

        return Dataset(id=dataset_id, description=description)

    def eval(self, cl4py_instruction: Tuple) -> Any:
        result = self._lisp_eval( cl4py_instruction )

        return result

    def all_compositions(self, dataset: Dataset) -> List[Composition]:
        descriptions = self._lisp_eval( ("mapcar", ("function", "idyom-db::composition-description"), ("idyom-db:get-compositions", dataset.id)) )
        composition_ids = self._lisp_eval( ("mapcan", ("function", ("lambda", ("x",), ("cdr", ("idyom-db:get-id", "x")))), ("idyom-db:get-compositions", dataset.id)) )
        result = list()
        for idx, description in enumerate(descriptions): # assuming that idx always coincides with the composition's id within the dataset
            result.append(Composition(dataset_id=dataset.id, id=composition_ids[idx], description=description))

        return result

    def derive_viewpoint_sequence(self, composition: Composition, viewpoints: List[Viewpoint]) -> List:
        cmd = ("viewpoints:viewpoint-sequence", (
        "viewpoints:get-viewpoint", ("quote", tuple(viewpoints_list_to_string_list(viewpoints)))),
                                              ("md:get-event-sequence", composition.dataset_id, composition.id))
        try:
            viewpoint_sequence = self._lisp_eval(cmd)
            viewpoint_sequence = cl4py_cons_to_list(viewpoint_sequence)
        except:
            print("Warning! Error while executing", cmd)
            viewpoint_sequence = []
        return viewpoint_sequence

    def get_alphabet(self, datasets, viewpoint: BasicViewpoint) -> List:
        if isinstance(datasets, Dataset):
            dataset_ids = [str(datasets.id)]
        elif isinstance(datasets, list):
            dataset_ids = list(map(lambda o: str(o.id), datasets))

        cmd = ["idyom-db::get-alphabet", ("quote", viewpoint.value), *dataset_ids]
        alphabet = self._lisp_eval( tuple(cmd) )
        alphabet = list(map(lambda x: x, alphabet))  # convert Cons to list
        return alphabet


def infer_target_viewpoints_target_viewpoint_values_and_used_source_viewpoints(fieldnames): # "used", because each target viewpoint may use only a subset of all provided source viewpoints
    unrelatedFieldnames = ['dataset.id', 'melody.id', 'note.id', 'melody.name', 'vertint12', 'articulation', 'comma',
                           'voice', 'ornament', 'dyn', 'phrase', 'bioi', 'deltast', 'accidental', 'mpitch', 'cpitch',
                           'barlength', 'pulses', 'tempo', 'mode', 'keysig', 'dur', 'onset',
                           'probability', 'information.content', 'entropy', 'information.gain',
                           '']
    remainingFieldnames = [o for o in fieldnames if o not in unrelatedFieldnames]

    targetViewpoints = list(set(list(map(lambda o: o.split(".", 1)[0], remainingFieldnames))))

    targetViewpointValues = dict()  # target viewpoint => list of values
    for tv in targetViewpoints:
        candidates = [o for o in remainingFieldnames if o.startswith(tv) and not any(
            target in o for target in ["weight", "ltm", "stm", "probability", "information.content", "entropy"])]
        values = list(map(lambda o: o.split(".")[1], candidates))  # note: leave it unsorted?
        targetViewpointValues[tv] = values

    usedSourceViewpoints = dict()  # target viewpoint => list of source viewpoints
    for tv in targetViewpoints:
        candidates = [o for o in remainingFieldnames if o.startswith(tv + ".order.stm.")]
        usedSourceViewpoints[tv] = list(set(list(map(lambda o: o.split(".")[3], candidates))))

    return targetViewpoints, targetViewpointValues, usedSourceViewpoints


class IDYOMResultsFile:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.targetViewpoints, self.targetViewpointValues, self.usedSourceViewpoints = infer_target_viewpoints_target_viewpoint_values_and_used_source_viewpoints(df.columns.values.tolist())


def parse_idyom_results(file_path: Path):
    df = pd.read_csv(file_path, sep=" ")

    unnamedColumns = df.columns.str.match("Unnamed")
    df = df.loc[:,~unnamedColumns] # remove unnamed columns

    return IDYOMResultsFile(df)
