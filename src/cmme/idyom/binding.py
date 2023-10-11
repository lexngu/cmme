from __future__ import annotations

from pathlib import Path
import pandas as pd
from typing import Union
import re

from .base import transform_viewpoints_list_to_string_list, IDYOMModelType, IDYOMViewpointSelectionBasis, \
    transform_string_list_to_viewpoints_list, IDYOMEscapeMethod
from .util import LispExpressionBuilder, LispExpressionBuilderMode, escape_path_string
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
(load (SB-IMPL::USERINIT-PATHNAME))
(start-idyom)

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
(load (SB-IMPL::USERINIT-PATHNAME))

(ql:quickload "trivial-features" :silent t)
(ql:quickload "clsql" :silent t)
(defun start-idyom ()
    (defvar *idyom-root* "{}")
    (ql:quickload "idyom" :silent t)
    (clsql:connect '("{}") :if-exists :old :database-type :sqlite3))
(start-idyom)

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
            if results_file_path is None else str(results_file_path)
        results_file_path = '"' + escape_path_string(results_file_path) + '"'
        idyom_root_path = escape_path_string(path_as_string_with_trailing_slash(instructions_file.idyom_root_path))
        idyom_database_path = escape_path_string(instructions_file.idyom_database_path)
        if instructions_file.idyom_root_path is None or instructions_file.idyom_database_path is None:
            file_contents = IDYOMInstructionsFile.INSTRUCTIONS_FILE_DEFAULT_TEMPLATE\
                .format(results_file_path, idyom_cmd, filename_cmd)
        else:
            file_contents = IDYOMInstructionsFile.INSTRUCTIONS_FILE_CUSTOM_ROOT_AND_DATABASE_TEMPLATE \
                .format(idyom_root_path, idyom_database_path, results_file_path, idyom_cmd, filename_cmd)

        with open(instructions_file_path, "w") as f:
            f.write(file_contents)


    @staticmethod
    def load(file_path: Union[str, Path]) -> IDYOMInstructionsFile:
        with open(file_path, "r") as f:
            instructions_file_content = f.read()

        def extract_lisp_command(cmd_name, str):
            start_index = str.find("(" + cmd_name)
            open_parentheses = 1
            end_index = start_index + 1
            while open_parentheses != 0:
                if str[end_index] == "(":
                    open_parentheses += 1
                elif str[end_index] == ")":
                    open_parentheses -= 1
                end_index += 1
            return str[start_index:end_index]

        def find_single_or_none(pattern, str, to_int=False, to_bool=False):
            result = re.search(pattern, str)
            if result:
                assert len(result.groups()) == 1
                if to_int:
                    return int(result.groups()[0])
                elif to_bool:
                    if result.groups()[0] == "t":
                        return True
                    elif result.groups()[0] == "nil":
                        return False
                    else:
                        raise ValueError("Could not determine bool value of {} (pattern: {})".format(result, pattern))
                else:
                    return result.groups()[0]
            else:
                return None

        def lisp_list_to_python_list(str, to_int=False, tuple_to_list=False):
            # (a, b, (c, d), e) => ["a", "b", ["c", "d"], "e"]
            if str is None:
                return None
            if not to_int:
                str_sub = re.sub(r"([^\s()]+)", r"'\1',", str)
            else:
                str_sub = re.sub(r"([^\s()]+)", r"\1,", str)
            if tuple_to_list:
                str_sub = re.sub(r"\(", r"[", str_sub)
                str_sub = re.sub(r"\)", r"]", str_sub)

            import ast
            res = ast.literal_eval(str_sub)

            return res

        idyom_root_path     = find_single_or_none(r"\(defvar\s+\*idyom-root\*\s+\"(.+)\"\)", instructions_file_content)
        idyom_database_path = find_single_or_none(r"\(clsql:connect '\(\"(.+)\"\)\s+:", instructions_file_content)
        output_dir          = find_single_or_none(r"\(defvar output-dir \"(.+)\"\)", instructions_file_content)

        # (idyom:idyom ...)
        idyom_cmd = extract_lisp_command("idyom:idyom", instructions_file_content)
        [dataset, search_target_viewpoints, search_source_viewpoints, models, search_stm_options,
         search_ltm_options] = re.search(
            r"\(idyom:idyom\s+(\d+)\s+'(\(.+?\))\s+(?:'(\(.+?\))|:select)(?:\s*:models\s+([^\s)]+))?(?:\s*:stmo\s+'(\(.+?\)))?(?:\s*:ltmo\s+'(\(.+?\)))?",
            idyom_cmd).groups()
        dataset = int(dataset)
        search_pretraining_ids = find_single_or_none(r":pretraining-ids\s+'(\(.+?\))", idyom_cmd)
        pretraining_ids = lisp_list_to_python_list(search_pretraining_ids, to_int=True, tuple_to_list=True)
        k = find_single_or_none(r":k\s+(\d+)", idyom_cmd, to_int=True)
        search_resampling_indices = find_single_or_none(r":resampling-indices\s+'(\(.+?\))", idyom_cmd)
        search_basis = find_single_or_none(r":basis\s+'?(.+)", idyom_cmd)
        dp = find_single_or_none(r":dp\s+(\d+)", idyom_cmd, to_int=True)
        max_links = find_single_or_none(r":max-links\s+(\d+)", idyom_cmd, to_int=True)
        min_links = find_single_or_none(r":min-links\s+(\d+)", idyom_cmd, to_int=True)
        viewpoint_selection_output = find_single_or_none(r":viewpoint-selection-output\s+\"(.+)\"",
                                                         idyom_cmd)
        detail      = find_single_or_none(r":detail\s+(\d+)", idyom_cmd, to_int=True)
        output_path = find_single_or_none(r":output-path\s+([^\s)]+|\"[^\s)]+\")", idyom_cmd)
        overwrite   = find_single_or_none(r":overwrite\s+([^\s)]+)", idyom_cmd, to_bool=True)
        separator   = find_single_or_none(r":separator\s+\"(.+)\"", idyom_cmd)
        use_resampling_set_cache = find_single_or_none(r":use-resampling-set-cache\?\s+([^\s)]+)",
                                                       idyom_cmd, to_bool=True)
        use_ltms_cache = find_single_or_none(r":use-ltms-cache\?\s+([^\s)]+)", idyom_cmd, to_bool=True)
        resampling_indices = lisp_list_to_python_list(search_resampling_indices, to_int=True, tuple_to_list=True)

        target_viewpoints = transform_string_list_to_viewpoints_list(lisp_list_to_python_list(search_target_viewpoints))
        source_viewpoints = transform_string_list_to_viewpoints_list(lisp_list_to_python_list(search_source_viewpoints))
        if search_stm_options:
            stm_order_bound = find_single_or_none(r":order-bound\s+(\d+)", search_stm_options, to_int=True)
            stm_mixtures = find_single_or_none(r":mixtures\s+([^\s)]+)", search_stm_options, to_bool=True)
            stm_update_exclusion = find_single_or_none(r":update-exclusion\s+([^\s)]+)", search_stm_options,
                                                       to_bool=True)
            stm_escape = find_single_or_none(r":escape\s+([^\s)]+)", search_stm_options)
            if stm_escape is not None:
                stm_escape = IDYOMEscapeMethod(stm_escape)
        else:
            stm_order_bound = stm_mixtures = stm_update_exclusion = stm_escape = None
        stm_options = {
            "order_bound": stm_order_bound,
            "mixtures": stm_mixtures,
            "update_exclusion": stm_update_exclusion,
            "escape": stm_escape
        }
        if search_ltm_options:
            ltm_order_bound = find_single_or_none(r":order-bound\s+(\d+)", search_ltm_options, to_int=True)
            ltm_mixtures = find_single_or_none(r":mixtures\s+([^\s)]+)", search_ltm_options, to_bool=True)
            ltm_update_exclusion = find_single_or_none(r":update-exclusion\s+([^\s)]+)", search_ltm_options,
                                                       to_bool=True)
            ltm_escape = find_single_or_none(r":escape\s+([^\s)]+)", search_ltm_options)
            if ltm_escape is not None:
                ltm_escape = IDYOMEscapeMethod(ltm_escape)
        else:
            ltm_order_bound = ltm_mixtures = ltm_update_exclusion = ltm_escape = None
        ltm_options = {
            "order_bound": ltm_order_bound,
            "mixtures": ltm_mixtures,
            "update_exclusion": ltm_update_exclusion,
            "escape": ltm_escape
        }

        training_options = {
            "pretraining_dataset_ids": pretraining_ids,
            "resampling_folds_count_k": k,
            "exclusively_to_be_used_resampling_fold_indices": resampling_indices
        }

        basis = lisp_list_to_python_list(search_basis)
        select_options = {
            "basis": basis,
            "dp": dp,
            "max_links": max_links,
            "min_links": min_links,
            "viewpoint_selection_output": viewpoint_selection_output
        }

        if output_path == "output-dir" and output_dir is None:
            raise ValueError(":output-path is set to value 'output-dir', but output-dir could not be determined!")
        output_options = {
            "output_path": output_path if output_path != "output-dir" else output_dir,
            "detail": detail,
            "overwrite": overwrite,
            "separator": separator
        }

        caching_options = {
            "use_resampling_set_cache": use_resampling_set_cache,
            "use_ltms_cache": use_ltms_cache
        }

        return IDYOMInstructionsFile(dataset, target_viewpoints, source_viewpoints, models,
                                     stm_options, ltm_options, training_options,
                                     select_options, output_options, caching_options,
                                     idyom_root_path, idyom_database_path)

    
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
        
    def _set_idyom_boilerplate(self, leb: LispExpressionBuilder, use_check_model_defaults=False):
        """
        Set shared values across IDyOM commands like (idyom:idyom ...). 
        
        Parameters
        ----------
        leb
            LispExpressionBuilder
        use_check_model_defaults
            Whether to use resampling::check-model-defaults (necessary for filename inference)
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
        if self.model == IDYOMModelType.STM or self.model == IDYOMModelType.BOTH or \
                self.model == IDYOMModelType.BOTH_PLUS:
            order_bound = self.stm_options["order_bound"]
            mixtures = bool_to_lisp(self.stm_options["mixtures"])
            update_exclusion = bool_to_lisp(self.stm_options["update_exclusion"])
            escape = self.stm_options["escape"].value if self.stm_options["escape"] else None

            stmo = []
            if order_bound is not None:
                stmo.extend([":order-bound", order_bound])
            if mixtures is not None:
                stmo.extend([":mixtures", mixtures])
            if update_exclusion is not None:
                stmo.extend([":update-exclusion", update_exclusion])
            if escape is not None:
                stmo.extend([":escape", escape])
            if len(stmo) > 0:
                leb.add(":stmo")
                if use_check_model_defaults:
                    leb2 = LispExpressionBuilder(leb._mode)
                    leb2.add("apply")\
                        .add("#'resampling::check-model-defaults")\
                        .add(LispExpressionBuilder(leb._mode)
                             .add("cons")
                             .add("mvs::*stm-params*")
                             .add_list(stmo))
                    leb.add(leb2)
                else:
                    leb.add_list(stmo)
        # (... <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...] ...)
        if self.model == IDYOMModelType.LTM or self.model == IDYOMModelType.BOTH or \
                self.model == IDYOMModelType.BOTH_PLUS or self.model == IDYOMModelType.LTM_PLUS:
            order_bound = self.ltm_options["order_bound"]
            mixtures = bool_to_lisp(self.ltm_options["mixtures"])
            update_exclusion = bool_to_lisp(self.ltm_options["update_exclusion"])
            escape = self.ltm_options["escape"].value if self.ltm_options["escape"] else None

            ltmo = []
            if order_bound is not None:
                ltmo.extend([":order-bound", order_bound])
            if mixtures is not None:
                ltmo.extend([":mixtures", mixtures])
            if update_exclusion is not None:
                ltmo.extend([":update-exclusion", update_exclusion])
            if escape is not None:
                ltmo.extend([":escape", escape])
            if len(ltmo) > 0:
                leb.add(":ltmo")
                if use_check_model_defaults:
                    leb2 = LispExpressionBuilder(leb._mode)
                    leb2.add("apply")\
                        .add("#'resampling::check-model-defaults")\
                        .add(LispExpressionBuilder(leb._mode)
                             .add("cons")
                             .add("mvs::*ltm-params*")
                             .add_list(ltmo))
                    leb.add(leb2)
                else:
                    leb.add_list(ltmo)
        # (... <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] ...)
        if self.training_options:
            if self.training_options["pretraining_dataset_ids"] is not None:
                pretraining_ids = list(map(str, self.training_options["pretraining_dataset_ids"]))
                leb.add(":pretraining-ids").add_list(pretraining_ids)
            k = self.training_options["resampling_folds_count_k"]
            if k is not None:
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
                leb.add(":basis").add_list(map(lambda e: e.value, basis))
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
                leb.add(":viewpoint-selection-output").add_path_string(selection_output)
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
            leb.add(":output-path")\
                .add_path_string(path_as_string_with_trailing_slash(results_file_path))
        else:
            if self.output_options["output_path"]:
                leb.add(":output-path")\
                    .add_path_string(path_as_string_with_trailing_slash(self.output_options["output_path"]))
            else:
                leb.add(":output-path").add("nil")
        overwrite = bool_to_lisp(self.output_options["overwrite"])
        if overwrite is not None:
            leb.add(":overwrite").add(overwrite)
        separator = self.output_options["separator"]
        if separator:
            leb.add(":separator").add_string(self.output_options["separator"])

        # (... [:use-resampling-set-cache? ... :use-ltms-cache? ...])
        use_resampling_set_cache = bool_to_lisp(self.caching_options["use_resampling_set_cache"])
        use_ltms_cache = bool_to_lisp(self.caching_options["use_ltms_cache"])
        if use_resampling_set_cache is not None:
            leb.add(":use-resampling-set-cache?").add(use_resampling_set_cache)
        if use_ltms_cache is not None:
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
        self._set_idyom_boilerplate(leb, use_check_model_defaults=True)

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

        target_viewpoints = transform_string_list_to_viewpoints_list(list(set(map(lambda o: o.split(".", 1)[0], remaining_fieldnames))))

        target_viewpoint_values = dict()  # target viewpoint => list of values
        for tv in target_viewpoints:
            tv_value = tv.value
            candidates = [o for o in remaining_fieldnames if o.startswith(tv_value) and not any(
                target in o for target in ["weight", "ltm", "stm", "probability", "information.content", "entropy"])]
            values = list(map(lambda o: o.split(".")[1], candidates))  # note: leave it unsorted?
            target_viewpoint_values[tv] = values

        used_source_viewpoints = dict()  # target viewpoint => list of source viewpoints
        for tv in target_viewpoints:
            tv_value = tv.value
            candidates = [o for o in remaining_fieldnames if o.startswith(tv_value + ".order.stm.")]
            used_source_viewpoints[tv] = list(set(list(map(lambda o: o.split(".")[3], candidates))))

        return target_viewpoints, target_viewpoint_values, used_source_viewpoints
