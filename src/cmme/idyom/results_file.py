from pathlib import Path
import pandas as pd

def inferTargetViewpointsTargetViewpointValuesAndUsedSourceViewpoints(fieldnames): # "used", because each target viewpoint may use only a subset of all provided source viewpoints
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
        self.targetViewpoints, self.targetViewpointValues, self.usedSourceViewpoints = inferTargetViewpointsTargetViewpointValuesAndUsedSourceViewpoints(df.columns.values.tolist())

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