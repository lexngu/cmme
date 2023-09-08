from __future__ import annotations

from pathlib import Path
import pandas as pd
from typing import Union

from .base import transform_viewpoints_list_to_string_list, IDYOMModelValue, IDYOMViewpointSelectionBasis
from .util import LispExpressionBuilder, LispExpressionBuilderMode
from ..lib.instructions_file import InstructionsFile
from ..lib.results_file import ResultsFile
from ..lib.util import path_as_string_with_trailing_slash


def bool_to_lisp(b: bool | None) -> str | None:
    """
    Return a string representation of the bool (or None)
    Parameters
    ----------
    b: bool
        Value to represent as string
    Returns
    -------
    String representation or None
    """
    if b is True:
        return "t"
    elif b is False:
        return "nil"
    else:
        return None

class IDYOMInstructionsFile(InstructionsFile):

    INSTRUCTIONS_FILE_DEFAULT_TEMPLATE = """;; Run IDyOM
(with-open-file (*standard-output* "/dev/null" :direction :output
                                   :if-exists :supersede)
    (load (SB-IMPL::USERINIT-PATHNAME))
    (start-idyom))

;; Run IDyOM
(defvar output-dir {}) 
{}

;; Return results file path
(defvar filename {})
(pprint (if output-dir
    (concatenate 'string "results_file_path=" output-dir filename)
    filename))
"""

    INSTRUCTIONS_FILE_CUSTOM_ROOT_AND_DATABASE_TEMPLATE = """;; Run IDyOM
(with-open-file (*standard-output* "/dev/null" :direction :output
                                   :if-exists :supersede)
    (load (SB-IMPL::USERINIT-PATHNAME)))
(with-open-file (*standard-output* "/dev/null" :direction :output
                                   :if-exists :supersede)
    (ql:quickload "clsql")
    (defun start-idyom ()
        (defvar *idyom-root* "{}")
        (ql:quickload "idyom")
        (clsql:connect '("{}") :if-exists :old :database-type :sqlite3))
    (start-idyom)
    )

;; Run IDyOM
(defvar output-dir {}) 
{}

;; Return results file path
(defvar filename {})
(pprint (if output-dir
    (concatenate 'string "results_file_path=" output-dir filename)
    filename))
"""

    @staticmethod
    def save(instructions_file: IDYOMInstructionsFile, instructions_file_path: Union[str, Path],
             results_file_path: Union[str, Path] = None):
        """
        Creates an instructions file

        Parameters
        ----------
        instructions_file
        instructions_file_path
        results_file_path
            Path to the *directory*, where IDyOM is supposed to store its results.
            Note that IDyOM determines the file name by itself. When running the instructions file,
            the full path to the results file will be returned.
        """
        idyom_cmd: str = instructions_file\
            .to_run_expression(results_file_path_variable="output-dir")  # See INSTRUCTIONS_FILE_TEMPLATE
        filename_cmd: str = instructions_file.to_filename_inference_expression()

        results_file_path = path_as_string_with_trailing_slash(instructions_file.output_options["output_path"]) \
            if results_file_path is None else results_file_path
        results_file_path = '"' + results_file_path + '"'
        if instructions_file.idyom_root_path is None or instructions_file.idyom_database_path is None:
            file_contents = IDYOMInstructionsFile.INSTRUCTIONS_FILE_DEFAULT_TEMPLATE\
                .format(results_file_path, idyom_cmd, filename_cmd)
        else:
            file_contents = IDYOMInstructionsFile.INSTRUCTIONS_FILE_CUSTOM_ROOT_AND_DATABASE_TEMPLATE \
                .format(str(instructions_file.idyom_root_path), str(instructions_file.idyom_database_path),
                        results_file_path, idyom_cmd, filename_cmd)

        with open(instructions_file_path, "w") as f:
            f.write(file_contents)


    @staticmethod
    def load(file_path: Union[str, Path]) -> IDYOMInstructionsFile:
        raise NotImplementedError
    
    def __init__(self, dataset, target_viewpoints, source_viewpoints, model, stm_options, ltm_options, training_options,
                 select_options, output_options, caching_options,
                 idyom_root_path, idyom_database_path):
        self.dataset = dataset
        self.target_viewpoints = target_viewpoints
        self.source_viewpoints = source_viewpoints
        self.model = model
        self.stm_options = stm_options
        self.ltm_options = ltm_options
        self.training_options = training_options
        self.select_options = select_options
        self.output_options = output_options
        self.caching_options = caching_options

        self.idyom_root_path = idyom_root_path
        self.idyom_database_path = idyom_database_path
        
    def _set_idyom_boilerplate(self, leb: LispExpressionBuilder):
        """
        Set shared values across IDyOM commands like (idyom:idyom ...). 
        
        Parameters
        ----------
        leb
            LispExpressionBuilder

        Returns
        -------

        """
        # (... <dataset-id> ...)
        leb.add(self.dataset.id)
        # (... <dataset-id> <target-viewpoints> ...)
        target_viewpoint_names = list(map(lambda v: v.value, self.target_viewpoints))
        leb.add_list(target_viewpoint_names)
        # (... <dataset-id> <target-viewpoints> <source-viewpoints> ...)
        if not self.select_options == {}:  # if not empty
            leb.add(":select")
        else:
            source_viewpoint_names = transform_viewpoints_list_to_string_list(self.source_viewpoints)
            leb.add_list(source_viewpoint_names)
        # (... <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> ...)
        leb.add(":models").add(self.model.value)
        # (CMD <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] ...)
        if self.model == IDYOMModelValue.STM or self.model == IDYOMModelValue.BOTH or \
                self.model == IDYOMModelValue.BOTH_PLUS:
            order_bound = self.stm_options["order_bound"]
            mixtures = bool_to_lisp(self.stm_options["mixtures"])
            update_exclusion = bool_to_lisp(self.stm_options["update_exclusion"])
            escape = self.stm_options["escape"].value if self.stm_options["escape"] else None

            stmo = []
            if order_bound:
                stmo.extend([":order-bound", order_bound])
            if mixtures:
                stmo.extend([":mixtures", mixtures])
            if update_exclusion:
                stmo.extend([":update-exclusion", update_exclusion])
            if escape:
                stmo.extend([":escape", escape])
            if len(stmo) > 0:
                leb.add(":stmo").add_list(stmo)
        # (... <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...] ...)
        if self.model == IDYOMModelValue.LTM or self.model == IDYOMModelValue.BOTH or \
                self.model == IDYOMModelValue.BOTH_PLUS or self.model == IDYOMModelValue.LTM_PLUS:
            order_bound = self.ltm_options["order_bound"]
            mixtures = bool_to_lisp(self.ltm_options["mixtures"])
            update_exclusion = bool_to_lisp(self.ltm_options["update_exclusion"])
            escape = self.ltm_options["escape"].value if self.ltm_options["escape"] else None

            ltmo = []
            if order_bound:
                ltmo.extend([":order-bound", order_bound])
            if mixtures:
                ltmo.extend([":mixtures", mixtures])
            if update_exclusion:
                ltmo.extend([":update-exclusion", update_exclusion])
            if escape:
                ltmo.extend([":escape", escape])
            if len(ltmo) > 0:
                leb.add(":ltmo").add_list(ltmo)
        # (... <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] ...)
        if self.training_options:
            if self.training_options["pretraining_dataset_ids"] is not None:
                pretraining_ids = list(map(str, self.training_options["pretraining_dataset_ids"]))
                leb.add(":pretraining-ids").add_list(pretraining_ids)
            k = self.training_options["resampling_folds_count_k"]
            if k:
                leb.add(":k").add(k)
            if self.training_options["exclusively_to_be_used_resampling_fold_indices"]:
                indices = list(map(str, self.training_options["exclusively_to_be_used_resampling_fold_indices"]))
                leb.add(":resampling-indices").add_list(indices)
        # (... <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] [:basis ... :dp ... :max-links ... :min-links ...
        # :viewpoint-selection-output ...] ...)
        if self.select_options:
            basis = self.select_options["basis"]
            if isinstance(basis, IDYOMViewpointSelectionBasis):
                leb.add(":basis").add(basis.value)
            elif isinstance(basis, list):
                leb.add(":basis").add_list(basis)
            else:
                raise ValueError("basis is invalid!")

            dp = self.select_options["dp"]
            max_links = self.select_options["max_links"]
            min_links = self.select_options["min_links"]
            selection_output = self.select_options["viewpoint_selection_output"]
            if dp:
                leb.add(":dp").add(dp)
            if max_links:
                leb.add(":max-links").add(max_links)
            if min_links:
                leb.add(":min-links").add(min_links)
            if selection_output:
                leb.add(":viewpoint-selection-output").add_string(selection_output)
        # (CMD <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] [:basis ... :dp ... :max-links ... :min-links ...
        # :viewpoint-selection-output ...] :detail ...)
        detail = self.output_options["detail"]
        if detail:
            leb.add(":detail").add(detail)
        
    def to_run_expression(self, results_file_path: Union[str, Path] = None,
                          results_file_path_variable: str = None,
                          leb_mode: LispExpressionBuilderMode = LispExpressionBuilderMode.LISP) -> Union[str, tuple]:
        """
        Return lisp expression which runs IDyOM.

        Parameters
        ----------
        results_file_path
            If set, this specifies the output path for IDyOM's results. Otherwise, a temporary directory is used.
            Only one of the  parameters results_file_path and results_file_path_variable must be set.
        results_file_path_variable
            If set, this specifies the variable, which contains the output directory path for IDyOM's results.
            Otherwise, see results_file_path. Only one of these two parameters must be set.
        leb_mode
            Depending on the choice, whether to return a lisp expression for ordinary LISP use (i.e., as str),
            or for cl4py (i.e., as tuple).

        Returns
        -------
        Union[str, tuple]
        """
        leb = LispExpressionBuilder(leb_mode)

        # (idyom:idyom ...)
        leb.add("idyom:idyom")

        # (... <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] [:basis ... :dp ... :max-links ... :min-links ...
        # :viewpoint-selection-output ...] :detail ...)
        self._set_idyom_boilerplate(leb)

        # (... [:output-path ... :overwrite ... :separator ...] ...)
        if results_file_path is not None and results_file_path_variable:
            raise ValueError("Either results_file_path or results_file_path_variable must be set.")

        if results_file_path_variable is not None:
            leb.add(":output-path").add(results_file_path_variable)
        elif results_file_path is not None:
            leb.add(":output-path").add_string(path_as_string_with_trailing_slash(results_file_path))
        else:
            if self.output_options["output_path"]:
                leb.add(":output-path").add_string(path_as_string_with_trailing_slash(self.output_options["output_path"]))
            else:
                leb.add(":output-path").add("nil")
        overwrite = bool_to_lisp(self.output_options["overwrite"])
        if overwrite:
            leb.add(":overwrite").add(overwrite)
        separator = self.output_options["separator"]
        if separator:
            leb.add(":separator").add_string(self.output_options["separator"])

        # (... [:use-resampling-set-cache? ... :use-ltms-cache? ...])
        use_resampling_set_cache = bool_to_lisp(self.caching_options["use_resampling_set_cache"])
        use_ltms_cache = bool_to_lisp(self.caching_options["use_ltms_cache"])
        if use_resampling_set_cache:
            leb.add(":use-resampling-set-cache?").add(use_resampling_set_cache)
        if use_ltms_cache:
            leb.add(":use-ltms-cache?").add(use_ltms_cache)

        return leb.build()

    def to_filename_inference_expression(self, leb_mode: LispExpressionBuilderMode = LispExpressionBuilderMode.LISP) \
            -> Union[str, tuple]:
        """
        Return lisp expression which fetches the filename of the IDyOM's results/output file.

        Parameters
        ----------

        leb_mode
            Depending on the choice, whether to return a lisp expression for ordinary LISP use (i.e., as str),
            or for cl4py (i.e., as tuple).

        Returns
        -------

        """
        leb = LispExpressionBuilder(leb_mode)

        # (apps:dataset-modelling-filename ...)
        leb.add("apps:dataset-modelling-filename")

        # (... <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] [:basis ... :dp ... :max-links ... :min-links ...
        # :viewpoint-selection-output ...] :detail ...)
        self._set_idyom_boilerplate(leb)

        # Set detail = 3 if not present (due to a bug of IDyOM, where this must set explicitly)
        if ":detail" not in leb.components:
            leb.add(":detail").add(3)

        # (... :extension)
        leb.add(":extension").add_string('.dat')

        return leb.build()


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
