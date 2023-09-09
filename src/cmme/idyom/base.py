import dataclasses
from enum import Enum
from typing import List


@dataclasses.dataclass
class Dataset:
    id: int
    description: str

    def __init__(self, id, description):
        self.id = id
        self.description = description


@dataclasses.dataclass
class Composition:
    dataset_id: int
    id: int
    description: str

    def __init__(self, dataset_id, id, description):
        self.dataset_id = dataset_id
        self.id = id
        self.description = description


class Viewpoint(Enum):
    pass


class BasicViewpoint(Viewpoint):
    ONSET = 'onset'
    CPITCH = 'cpitch'
    DUR = 'dur'
    KEYSIG = 'keysig'
    MODE = 'mode'
    TEMPO = 'tempo'
    PULSES = 'pulses'
    BARLENGTH = 'barlength'
    DELTAST = 'deltast'
    BIOI = 'bioi'
    PHRASE = 'phrase'

    MPITCH = 'mpitch'
    ACCIDENTAL = 'accidental'
    DYN = 'dyn'
    VOICE = 'voice'
    ORNAMENT = 'ornament'
    COMMA = 'comma'
    ARTICULATION = 'articulation'


class DerivedViewpoint(Viewpoint):
    # based on onset:
    IOI = 'ioi'
    POSINBAR = 'posinbar'
    # based on dur:
    DUR_RATIO = 'dur-ratio'
    # based on keysig:
    REFERENT = 'referent'
    # based on cpitch:
    CPINT = 'cpint'
    CONTOUR = 'contour'
    CPITCH_CLASS = 'cpitch-class'
    CPCINT = 'cpcint'
    CPINTREF = 'cpintref'
    CPINTFIP = 'cpintfip'
    CPINTFIPH = 'cpintfiph'
    CPINTFIB = 'cpintfib'
    INSCALE = 'inscale'

    # based on onset:
    IOI_RATIO = 'ioi-ratio'
    IOI_CONTOUR = 'ioi-contour'
    METACCENT = 'metaccent'
    # based on bioi:
    BIOI_RATIO = 'bioi-ratio'
    BIOI_CONTOUR = 'bioi-contour'
    # based on phrase:
    LPHRASE = 'lphrase'
    # based on cpitch:
    CPINT_SIZE = 'cpint-size'
    NEWCONTOUR = 'newcontour'
    CPCINT_SIZE = 'cpcint-size'
    CPCINT_2 = 'cpcint-2'
    CPCINT_3 = 'cpcint-3'
    CPCINT_4 = 'cpcint-4'
    CPCINT_5 = 'cpcint-5'
    CPCINT_6 = 'cpcint-6'
    OCTAVE = 'octave'
    TESSITURA = 'tessitura'
    # based on mpitch:
    MPITCH_CLASS = 'mpitch-class'

    # based on cpitch:
    REGISTRAL_DIRECTION = 'registral-direction'
    INTERVALLIC_DIFFERENCE = 'intervallic-difference'
    REGISTRAL_RETURN = 'registral-return'
    PROXIMITY = 'proximity'
    CLOSURE = 'closure'


class TestViewpoint(Viewpoint):
    FIB = 'fib'
    CROTCHET = 'crotchet'
    TACTUS = 'tactus'
    FIPH = 'fiph'
    LIPH = 'liph'


class ThreadedViewpoint(Viewpoint):
    # based on cpitch and onset:
    THR_CPINT_FIB = 'thr-cpint-fib'
    THR_CPINT_FIPH = 'thr-cpint-fiph'
    THR_CPINT_LIPH = 'thr-cpint-liph'
    THR_CPINT_CROTCHET = 'thr-cpint-crotchet'
    THR_CPINT_TACTUS = 'thr-cpint-tactus'
    THR_CPINTREF_LIPH = 'thr-cpintref-liph'
    THR_CPINTREF_FIB = 'thr-cpintref-fib'
    THR_CPINT_CPINTREF_LIPH = 'thr-cpint_cpintref-liph'
    THR_CPINT_CPINTREF_FIB = 'thr-cpint_cpintref-fib'


class IDYOMModelValue(Enum):
    STM = ':stm'
    LTM = ':ltm'
    LTM_PLUS = ':ltm+'
    BOTH = ':both'
    BOTH_PLUS = ':both+'


class IDYOMEscape(Enum):
    A = ':a'
    B = ':b'
    C = ':c'
    D = ':d'
    X = ':x'


class IDYOMViewpointSelectionBasis(Enum):
    AUTO = ':auto'
    PITCH_FULL = ':pitch-full'
    PITCH_SHORT = ':pitch-short'
    BIOI = ':bioi'
    ONSET = ':onset'


def transform_viewpoints_list_to_string_list(viewpoints: List[Viewpoint]) -> List[str]:
    """
    Transform a (possibly nested) list of viewpoints to a list of strings, where
    each string corresponds to the string-name of each viewpoint.

    Parameters
    ----------
    viewpoints
        List of viewpoints

    Returns
    -------
    List[str]
        List of strings
    """
    result = []

    if isinstance(viewpoints, Viewpoint):
        result.append(viewpoints.value)
    elif isinstance(viewpoints, list) or isinstance(viewpoints, tuple):
        if len(viewpoints) == 0:
            raise ValueError("viewpoints invalid! Length must be greater than zero.")
        for viewpoint in viewpoints:
            recursion_result = transform_viewpoints_list_to_string_list(viewpoint)
            if len(recursion_result) == 1:
                result.append(recursion_result[0])
            else:
                result.append(recursion_result)
    else:
        raise ValueError("Invalid element: " + str(viewpoints))

    return result

def transform_string_list_to_viewpoints_list(viewpoints: List[str]) -> List[Viewpoint]:
    """
    Transform a (possibly nested) list of viewpoint-strings to a list of corresponding viewpoints.

    Parameters
    ----------
    viewpoints
        List of viewpoint-strings

    Returns
    -------
    List[str]
        List of viewpoints
    """
    result = []

    if isinstance(viewpoints, str):
        viewpoint = None
        for cls in [BasicViewpoint, DerivedViewpoint, TestViewpoint, ThreadedViewpoint]:
            try:
                viewpoint = cls(viewpoints)
                break
            except:
                pass
        if viewpoint is not None:
            result.append(viewpoint)
        else:
            raise ValueError("Could not determine corresponding enum for viewpoint {}".format(viewpoints))
    elif isinstance(viewpoints, list) or isinstance(viewpoints, tuple):
        if len(viewpoints) == 0:
            raise ValueError("viewpoints invalid! Length must be greater than zero.")
        for viewpoint in viewpoints:
            recursion_result = transform_string_list_to_viewpoints_list(viewpoint)
            if len(recursion_result) == 1:
                result.append(recursion_result[0])
            else:
                result.append(recursion_result)
    else:
        raise ValueError("Invalid element: " + str(viewpoints))

    return result