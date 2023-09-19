from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Union, Tuple

import cl4py
from cl4py import Cons

from cmme.idyom.base import Viewpoint, BasicViewpoint, DerivedViewpoint, ThreadedViewpoint, TestViewpoint
from cmme.lib.util import path_as_string_with_trailing_slash

import subprocess


def escape_path_string(path) -> str:
    return str(path).replace("\\", "\\\\")

class LispExpressionBuilderMode(Enum):
    CL4PY = "cl4py",
    LISP = "lisp"


class LispExpressionBuilder:
    def __init__(self, mode: LispExpressionBuilderMode):
        self.components = None
        self._mode = mode

        self.reset()

    def reset(self):
        self.components = []
        return self

    def add(self, value):
        value = str(value) if not isinstance(value, LispExpressionBuilder) else value.build()
        self.components.append(value)
        return self

    def add_string(self, value):
        value = str(value)
        self.components.append('"' + value + '"')
        return self

    def add_path_string(self, value):
        value = str(escape_path_string(value))
        self.components.append('"' + value + '"')
        return self

    def _as_nested_list_string(self, value):
        result = []
        if not (isinstance(value, list) or isinstance(value, tuple)):
            result.append(value)
        else:
            for e in value:
                if isinstance(e, list) or isinstance(e, tuple):
                    result.append(self._as_nested_list_string(e)[1:])  # i.e. without '
                else:
                    result.append(str(e))
        return "'(" + " ".join(result) + ")"

    def _as_nested_tuple(self, value):
        if not (isinstance(value, list) or isinstance(value, tuple)):
            value = [value]
        result = []
        for e in value:
            if isinstance(e, list) or isinstance(e, tuple):
                result.append(self._as_nested_tuple(e))
            else:
                result.append(e)
        return tuple(result)

    def add_list(self, value):
        if self._mode == LispExpressionBuilderMode.LISP:
            self.components.append(self._as_nested_list_string(value))
        elif self._mode == LispExpressionBuilderMode.CL4PY:
            self.components.append(('quote', self._as_nested_tuple(value)))
        return self

    def build(self) -> Union[str, tuple]:
        if self._mode == LispExpressionBuilderMode.LISP:
            return str(self)
        elif self._mode == LispExpressionBuilderMode.CL4PY:
            return tuple(self.components)

    def __str__(self):
        return "(" + " ".join(self.components) + ")"


def cl4py_cons_to_list(cons):
    """
    Transforms cl4py's "Cons" to a Python list.
    List(1) => [1]
    List(List(1, 2), List(3, 4)) => [[1, 2], [3, 4]]
    :param cons:
    :return:
    """
    result = []
    if isinstance(cons, Cons):
        for e in cons:
            recursion_result = cl4py_cons_to_list(e)
            if isinstance(recursion_result, list) and len(recursion_result) == 1:
                result.append(recursion_result[0])
            else:
                result.append(recursion_result)
    else:
        result = cons
    return result


def install_idyom(idyom_root_path: Union[str, Path], idyom_database_path: Union[str, Path]=None, force_reset=False) -> Path:
    """
    Installs idyom as quicklisp project, and initializes the database
    :param idyom_root_path:
    :param idyom_database_path
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
    idyom_db_path = Path(os.path.join(idyom_root_path, "db/database.sqlite")) if idyom_database_path is None \
        else Path(idyom_database_path)

    if idyom_db_path.exists() and not force_reset:
        print(
            "Database at {} already exists. Use force_reset, if you want to reset the database.".format(idyom_db_path))

    if not idyom_db_path.parent.exists():
        Path(idyom_db_path).parent.mkdir(parents=True)

    lisp.eval(('defvar', 'common-lisp-user::*idyom-root*', '"' + escape_path_string(idyom_root_path) + '"'))
    lisp.eval(('ql:quickload', '"idyom"'))
    lisp.eval(('clsql:connect', (
        'list', '"' + escape_path_string(idyom_db_path) + '"'), ':if-exists', ':old', ':database-type', ':sqlite3'))
    lisp.eval(('idyom-db:initialise-database',))

    return idyom_db_path


def run_idyom_instructions_file(instructions_file_path: Union[str, Path]) -> Tuple[str, str]:
    if not os.path.exists(instructions_file_path):
        raise ValueError("instructions_file_path points to a non-existing file!")
    process = subprocess.Popen(["sbcl", "--script", instructions_file_path],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    out, err = process.communicate()
    return out, err


def viewpoint_name_to_viewpoint(name: str) -> Viewpoint:
    """
    Return the associated viewpoint object.

    Parameters
    ----------
    name
        Viewpoint name

    Returns
    -------
    Viewpoint
        Viewpoint object, if viewpoint could be determined. ValueError otherwise.
    """
    candidates = [BasicViewpoint, DerivedViewpoint, ThreadedViewpoint, TestViewpoint]
    result = None
    for candidate in candidates:
        try:
            result = candidate(name)
        except ValueError:
            pass

    if result is None:
        raise ValueError("Viewpoint with name={} invalid!".format(name))

    return result
