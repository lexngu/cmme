from enum import Enum


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
        self.components.append(str(value))
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
