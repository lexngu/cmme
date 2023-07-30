from __future__ import annotations

from pathlib import Path
import pandas as pd
from typing import Union
from ..lib.results_file import ResultsFile


class IDYOMResultsFile(ResultsFile):

    @staticmethod
    def save(results_file: ResultsFile, file_path: Union[str, Path]):
        raise NotImplementedError

    @staticmethod
    def load(file_path: Union[str, Path]) -> IDYOMResultsFile:
        df = pd.read_csv(file_path, sep=" ")

        unnamed_columns = df.columns.str.match("Unnamed")
        df = df.loc[:, ~unnamed_columns]  # remove unnamed columns

        return IDYOMResultsFile(df)

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        self.targetViewpoints, self.targetViewpointValues, self.usedSourceViewpoints = \
            self.infer_target_viewpoints_target_viewpoint_values_and_used_source_viewpoints(df.columns.values.tolist())

    @staticmethod
    def infer_target_viewpoints_target_viewpoint_values_and_used_source_viewpoints(
            fieldnames):  # "used", because each target viewpoint may use only a subset of all provided source viewpoints
        unrelated_fieldnames = ['dataset.id', 'melody.id', 'note.id', 'melody.name', 'vertint12', 'articulation',
                                'comma',
                                'voice', 'ornament', 'dyn', 'phrase', 'bioi', 'deltast', 'accidental', 'mpitch',
                                'cpitch',
                                'barlength', 'pulses', 'tempo', 'mode', 'keysig', 'dur', 'onset',
                                'probability', 'information.content', 'entropy', 'information.gain',
                                '']
        remaining_fieldnames = [o for o in fieldnames if o not in unrelated_fieldnames]

        target_viewpoints = list(set(list(map(lambda o: o.split(".", 1)[0], remaining_fieldnames))))

        target_viewpoint_values = dict()  # target viewpoint => list of values
        for tv in target_viewpoints:
            candidates = [o for o in remaining_fieldnames if o.startswith(tv) and not any(
                target in o for target in ["weight", "ltm", "stm", "probability", "information.content", "entropy"])]
            values = list(map(lambda o: o.split(".")[1], candidates))  # note: leave it unsorted?
            target_viewpoint_values[tv] = values

        used_source_viewpoints = dict()  # target viewpoint => list of source viewpoints
        for tv in target_viewpoints:
            candidates = [o for o in remaining_fieldnames if o.startswith(tv + ".order.stm.")]
            used_source_viewpoints[tv] = list(set(list(map(lambda o: o.split(".")[3], candidates))))

        return target_viewpoints, target_viewpoint_values, used_source_viewpoints
