import dataclasses
import pandas as pd
from enum import Enum
from pathlib import Path
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

def viewpoints_list_as_lisp_string(viewpoints: List[Viewpoint]) -> str:
    result = ""

    if isinstance(viewpoints, Viewpoint):
        return viewpoints.value
    elif isinstance(viewpoints, list) or isinstance(viewpoints, tuple):
        if len(viewpoints) == 0:
            result += "()"
        else:
            result += " ("
            for viewpoint in viewpoints:
                result += viewpoints_list_as_lisp_string(viewpoint) + " "
            result = result[:-1] # remove last character
            result += ")"
    else:
        raise ValueError("Invalid element: " + str(viewpoints))

    return result.strip()

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


# def parse_idyom_results(file_path: Path):
#     with open(file_path, 'r') as f:
#         csvreader = csv.DictReader(f, delimiter=" ")
#         targetViewpoints, targetViewpointValues, usedSourceViewpoints = inferTargetViewpointsTargetViewpointValuesAndUsedSourceViewpoints(
#             csvreader.fieldnames)
#
#         for row in csvreader:
#             # database identifiers
#             datasetId = row['dataset.id']
#             melodyId = row['melody.id']
#             noteId = row['note.id']
#             melodyName = row['melody.name']
#             vertint12 = row['vertint12']  # undocumented?
#             # musical properties of the event
#             articulation = row['articulation']
#             comma = row['comma']
#             voice = row['voice']
#             ornament = row['ornament']
#             dyn = row['dyn']
#             phrase = row['phrase']
#             bioi = row['bioi']
#             deltast = row['deltast']
#             accidental = row['accidental']
#             mpitch = row['mpitch']
#             cpitch = row['cpitch']
#             barlength = row['barlength']
#             pulses = row['pulses']
#             tempo = row['tempo']
#             mode = row['mode']
#             keysig = row['keysig']
#             dur = row['dur']
#             onset = row['onset']
#
#             # model properties
#             ltmOrder = dict()  # target viewpoint => (source viewpoint => int)
#             stmOrder = dict()  # target viewpoint => (source viewpoint => int)
#             ltmWeight = dict()  # target viewpoint => int
#             stmWeight = dict()  # target viewpoint => int
#             ltmWeightBySourceViewpoint = dict()  # target viewpoint => (source viewpoint => int)
#             stmWeightBySourceViewpoint = dict()  # target viewpoint => (source viewpoint => int)
#
#             # model output
#             targetViewpointProbability = dict()  # targetViewpoint => probability
#             targetViewpointInformationContent = dict()  # targetViewpoint => IC
#             targetViewpointEntropy = dict()  # targetViewpoint => E
#             targetViewpointProbabilityDistribution = dict()  # targetViewpoint => dict(x => p(x))
#
#             # overall probability, IC and E
#             probability = row['probability']
#             informationContent = row['information.content']
#             entropy = row['entropy']
#             informationGain = row['information.gain']  # undocumented?
#
#             for tv in targetViewpoints:
#                 ltmWeight[tv] = row[tv + ".weight.ltm"]
#                 stmWeight[tv] = row[tv + ".weight.stm"]
#
#                 targetViewpointProbability[tv] = row[tv + ".probability"]
#                 targetViewpointInformationContent[tv] = row[tv + ".information.content"]
#                 targetViewpointEntropy[tv] = row[tv + ".entropy"]
#                 targetViewpointProbabilityDistribution[tv] = list(v for k, v in row.items() if any(k == (tv + "." + target) for target in targetViewpointValues[tv]))
#
#                 for sv in usedSourceViewpoints[tv]:
#                     ltmOrder[tv] = dict()
#                     ltmOrder[tv][sv] = row[tv + ".order.ltm." + sv]
#
#                     stmOrder[tv] = dict()
#                     stmOrder[tv][sv] = row[tv + ".order.stm." + sv]
#
#                     ltmWeightBySourceViewpoint[tv] = dict()
#                     ltmWeightBySourceViewpoint[tv][sv] = row[tv + ".weight.ltm." + sv]
#
#                     stmWeightBySourceViewpoint[tv] = dict()
#                     stmWeightBySourceViewpoint[tv][sv] = row[tv + ".weight.stm." + sv]