from datetime import datetime
from enum import Enum
from pathlib import Path

from cl4py import Cons

from cmme.config import Config


class LispExpressionBuilderMode(Enum):
    CL4PY = "cl4py",
    LISP = "lisp"


class LispExpressionBuilder:
    def __init__(self, mode: LispExpressionBuilderMode):
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

    def _as_nested_list_string(self, value):
        result = []
        if not (isinstance(value, list) or isinstance(value, tuple)):
            result.append(value)
        else:
            for e in value:
                if isinstance(e, list) or isinstance(e, tuple):
                    result.append(self._as_nested_list_string(e)[1:]) # i.e. without '
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

    def build(self):
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


def idyom_default_instructions_file_path(alias: str = None, io_path: Path = Config().model_io_path()):
    instructions_file_filename = "idyom-instructionsfile"
    instructions_file_filename = (instructions_file_filename + "-" + alias) if alias is not None else instructions_file_filename
    instructions_file_filename = instructions_file_filename + ".csv"
    return io_path / instructions_file_filename
