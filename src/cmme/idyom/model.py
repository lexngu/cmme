import tempfile
import os
from .base import *
from .binding import IDYOMBinding
from cmme.util import flatten_list
from ..config import Config
from .util import *


class IDYOMInstructionBuilder:
    def __init__(self):
        self._dataset: Dataset = None
        self._target_viewpoints: List[Viewpoint] = []
        self._source_viewpoints: List[Viewpoint] = []
        self._model: IDYOMModelValue = IDYOMModelValue.BOTH_PLUS

        self.stm_options()
        self.ltm_options()
        self.training_options()
        self.automatically_select_source_viewpoints()
        self.output_options()
        self.caching_options()

    def dataset(self, dataset_or_id):
        if isinstance(dataset_or_id, Dataset):
            self._dataset = dataset_or_id
        elif isinstance(dataset_or_id, int):
            self._dataset = Dataset(id=dataset_or_id, description="(Unchecked) id value")
        else:
            raise ValueError("Dataset invalid! Provide an id value or Dataset instance.")
        return self

    def target_viewpoints(self, target_viewpoints: List[BasicViewpoint]):
        if any(not isinstance(v, BasicViewpoint) for v in target_viewpoints):
            raise ValueError("target_viewpoints must not contain anything but BasicViewpoint!")
        self._target_viewpoints = target_viewpoints
        return self

    def source_viewpoints(self, source_viewpoints: List[Viewpoint]):
        """
        Sets the source viewpoints. If you want to use IDyOM's automatic selection algorithm, ignore this function. Instead use #automatically_select_source_viewpoints(...)
        :param source_viewpoints:
        :return:
        """
        if any(not isinstance(v, Viewpoint) or isinstance(v, List) for v in flatten_list(source_viewpoints, True)):
            raise ValueError("source_viewpoints must not contain anything but (potentially nested) Viewpoints!")
        self._source_viewpoints = source_viewpoints
        return self

    def model(self, model: IDYOMModelValue):
        self._model = model
        return self

    def stm_options(self, order_bound: int = None, mixtures=True, update_exclusion=True, escape=IDYOMEscape.X): # original default values
        self._stm_options = {
            "order_bound": order_bound,
            "mixtures": mixtures,
            "update_exclusion": update_exclusion,
            "escape": escape
        }
        return self

    def ltm_options(self, order_bound = None, mixtures = True, update_exclusion = False, escape = IDYOMEscape.C): # original default values
        self._ltm_options = {
            "order_bound": order_bound,
            "mixtures": mixtures,
            "update_exclusion": update_exclusion,
            "escape": escape
        }
        return self

    def training_options(self, pretraining_dataset_ids = None, resampling_folds_count_k = 10, exclusively_to_be_used_resampling_fold_indices = None):
        """
        Training options affect all model configurations (STM, LTM, LTM+, BOTH, BOTH+).
        If STM, there is no training, however, the pretraining datasets provide the "viewpoint domain" (i.e. PPM's "alphabet") for the STM model.
        For all the remaining configurations (LTM, ...), this pre-training happens before IDyOM's resampling procedure.
        :param pretraining_dataset_ids: If None, training options are reset.
        :param resampling_folds_count_k:
        :param exclusively_to_be_used_resampling_fold_indices: If None, IDyOM will use all folds
        :return:
        """
        if pretraining_dataset_ids is None:
            self._training_options = {}
        else:
            self._training_options = {
                "pretraining_dataset_ids": pretraining_dataset_ids,
                "resampling_folds_count_k": resampling_folds_count_k,
                "exclusively_to_be_used_resampling_fold_indices": exclusively_to_be_used_resampling_fold_indices
            }
        return self

    def automatically_select_source_viewpoints(self, basis: IDYOMViewpointSelectionBasis = None, dp = None, max_links = 2, min_links = 2, viewpoint_selection_output = None):
        """

        :param basis: If None, select options are reset.
        :param dp:
        :param max_links:
        :param min_links:
        :param viewpoint_selection_output:
        :return:
        """
        if basis == None:
            self._select_options = {}
        else:
            self._select_options = {
                "basis": basis,
                "dp": dp,
                "max_links": max_links,
                "min_links": min_links,
                "viewpoint_selection_output": viewpoint_selection_output
            }
        return self

    def output_options(self, output_path = tempfile.gettempdir(), detail = 3, overwrite = False, separator =" "):
        if output_path is None:
            raise ValueError("None is not an allowed value for output_path!")

        self._output_options = {
            "output_path": output_path,
            "detail": detail,
            "overwrite": overwrite,
            "separator": separator
        }
        return self

    def caching_options(self, use_resampling_set_cache = True, use_ltms_cache = True):
        self._caching_options = {
            "use_resampling_set_cache": use_resampling_set_cache,
            "use_ltms_cache": use_ltms_cache
        }

    def _is_valid(self) -> tuple:
        """
        Checks the instruction builder for validity.
        :return: (is_valid: bool, error_msgs: list)
        """
        msgs = []
        is_valid = True

        if self._dataset is None:
            msgs.append("dataset must not be None!")
            is_valid = False
        if len(self._target_viewpoints) == 0:
            msgs.append("There must be at least one element in target_viewpoints!")
            is_valid = False
        if len(self._source_viewpoints) == 0:
            msgs.append("There must be at least one element in source_viewpoints!")
            is_valid = False
        if not isinstance(self._model, IDYOMModelValue):
            msgs.append("model must be any value of IDYOMModelValue!")

        return (is_valid, msgs)

    def assert_is_valid(self):
        builder_is_valid, builder_error_msgs = self._is_valid()
        if not builder_is_valid:
            raise ValueError(" ".join(builder_error_msgs))
        return

    def _idyom_boilerplate(self, leb: LispExpressionBuilder):
        # (CMD <dataset-id> ...)
        leb.add(self._dataset.id)
        # (CMD <dataset-id> <target-viewpoints> ...)
        target_viewpoint_names = list(map(lambda v: v.value, self._target_viewpoints))
        leb.add_list(target_viewpoint_names)
        # (CMD <dataset-id> <target-viewpoints> <source-viewpoints> ...)
        if not self._select_options == {}:  # if not empty
            leb.add(":select")
        else:
            source_viewpoint_names = viewpoints_list_to_string_list(self._source_viewpoints)
            leb.add_list(source_viewpoint_names)
        # (CMD <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> ...)
        leb.add(":models").add(self._model.value)
        # (CMD <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] ...)
        if self._model == IDYOMModelValue.STM or self._model == IDYOMModelValue.BOTH or self._model == IDYOMModelValue.BOTH_PLUS:
            order_bound = str(self._stm_options["order_bound"]) if self._stm_options[
                                                                       "order_bound"] is not None else "nil"
            mixtures = "t" if self._stm_options["mixtures"] else "nil"
            update_exclusion = "t" if self._stm_options["update_exclusion"] else "nil"
            escape = self._stm_options["escape"].value
            leb.add(":stmo").add_list(
                [":order-bound", order_bound, ":mixtures", mixtures, ":update-exclusion", update_exclusion, ":escape",
                 escape])
        # (CMD <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...] ...)
        if self._model == IDYOMModelValue.LTM or self._model == IDYOMModelValue.BOTH or self._model == IDYOMModelValue.BOTH_PLUS:
            order_bound = str(self._ltm_options["order_bound"]) if self._ltm_options[
                                                                       "order_bound"] is not None else "nil"
            mixtures = "t" if self._ltm_options["mixtures"] else "nil"
            update_exclusion = "t" if self._ltm_options["update_exclusion"] else "nil"
            escape = self._ltm_options["escape"].value
            leb.add(":ltmo").add_list(
                [":order-bound", order_bound, ":mixtures", mixtures, ":update-exclusion", update_exclusion, ":escape",
                 escape])
        # (CMD <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] ...)
        if self._training_options:
            pretraining_ids = list(map(str, self._training_options["pretraining_dataset_ids"]))
            leb.add(":pretraining-ids").add_list(pretraining_ids)
            leb.add(":k").add(self._training_options["resampling_folds_count_k"])
            if self._training_options["exclusively_to_be_used_resampling_fold_indices"]:
                indices = list(map(str, self._training_options["exclusively_to_be_used_resampling_fold_indices"]))
                leb.add(":resampling-indices").add_list(indices)
        # (CMD <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] [:basis ... :dp ... :max-links ... :min-links ... :viewpoint-selection-output ...] ...)
        if self._select_options:
            basis = self._select_options["basis"]
            if isinstance(basis, IDYOMViewpointSelectionBasis):
                leb.add(":basis").add(basis.value)
            elif isinstance(basis, list):
                leb.add(":basis").add_list(basis)
            else:
                print("ERROR")

            dp = str(self._select_options["dp"]) if self._select_options["dp"] is not None else "nil"
            leb.add(":dp").add(dp)
            leb.add(":max-links").add(self._select_options["max_links"])
            leb.add(":min-links").add(self._select_options["min_links"])
            leb.add(":viewpoint-selection-output").add_string(self._select_options["viewpoint_selection_output"])
        # (CMD <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] [:basis ... :dp ... :max-links ... :min-links ... :viewpoint-selection-output ...]
        # :detail ...)
        leb.add(":detail").add(self._output_options["detail"])

    def _build_idyomidyom_lispexpression(self, mode: LispExpressionBuilderMode) -> LispExpressionBuilder:
        leb = LispExpressionBuilder(mode)

        # (idyom:idyom ...)
        leb.add("idyom:idyom")

        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] [:basis ... :dp ... :max-links ... :min-links ... :viewpoint-selection-output ...]
        # :detail ...)
        self._idyom_boilerplate(leb)

        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] [:basis ... :dp ... :max-links ... :min-links ... :viewpoint-selection-output ...]
        # :detail [:output-path ... :overwrite ... :separator ...] ...)
        if self._output_options["output_path"]:
            leb.add(":output-path").add_string(self._output_options["output_path"])
        else:
            leb.add(":output-path").add("nil")
        leb.add(":overwrite").add("t" if self._output_options["overwrite"] else "nil")
        leb.add(":separator").add_string(self._output_options["separator"])
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] [:basis ... :dp ... :max-links ... :min-links ... :viewpoint-selection-output ...]
        # [:output-path ... :detail ... :overwrite ... :separator ...] [:use-resampling-set-cache? ... :use-ltms-cache? ...])
        leb.add(":use-resampling-set-cache?").add("t" if self._caching_options["use_resampling_set_cache"] else "nil")
        leb.add(":use-ltms-cache?").add("t" if self._caching_options["use_ltms_cache"] else "nil")

        return leb.build()

    def build_for_cl4py(self) -> tuple:
        self.assert_is_valid()
        return self._build_idyomidyom_lispexpression(LispExpressionBuilderMode.CL4PY)

    def build_for_lisp(self) -> str:
        self.assert_is_valid()
        return self._build_idyomidyom_lispexpression(LispExpressionBuilderMode.LISP)

    def build_for_cl4py_filename_inference(self) -> tuple:
        self.assert_is_valid()
        leb = LispExpressionBuilder(LispExpressionBuilderMode.CL4PY)

        # (apps:dataset-modelling-filename ...)
        leb.add("apps:dataset-modelling-filename")

        # (apps:dataset-modelling-filename <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] [:basis ... :dp ... :max-links ... :min-links ... :viewpoint-selection-output ...]
        # :detail ...)
        self._idyom_boilerplate(leb)

        # extension
        leb.add(":extension").add_string('.dat')

        return leb.build()

class IDYOMModel:
    def __init__(self, idyom_root_path: Path = Config().idyom_root_path(), idyom_database_path: Path = Config().idyom_database_path()):
        self.idyom_binding = IDYOMBinding(str(idyom_root_path.resolve()), str(idyom_database_path.resolve()))

    def import_midi(self, midi_files_directory_path: str, description: str, dataset_id: int = None, timebase: int = 96) -> Dataset:
        """

        :param midi_files_directory_path:
        :param description:
        :param dataset_id: If None, a valid value will be determined automatically
        :param timebase: Value of "kern2db::*default-timebase*" to use, MCCC requires 39473280.
        :return:
        """
        if dataset_id is None:
            dataset_id = self.idyom_binding.next_free_dataset_id()

        return self.idyom_binding.import_midi(midi_files_directory_path, description, dataset_id, timebase)

    def import_kern(self, krn_files_directory_path: str, description: str, dataset_id: int = None, timebase: int = 96) -> Dataset:
        """

        :param krn_files_directory_path:
        :param description:
        :param dataset_id: If None, a valid value will be determined automatically
        :param timebase: Value of "kern2db::*default-timebase*" to use, MCCC requires 39473280.
        :return:
        """
        if dataset_id is None:
            dataset_id = self.idyom_binding.next_free_dataset_id()

        return self.idyom_binding.import_kern(krn_files_directory_path, description, dataset_id, timebase)

    def all_datasets(self) -> List[Dataset]:
        return self.idyom_binding.all_datasets()

    def run(self, instruction_builder: IDYOMInstructionBuilder) -> IDYOMResultsFile:
        self.idyom_binding.eval( instruction_builder.build_for_cl4py() )

        filename = os.path.join(instruction_builder._output_options["output_path"], self.idyom_binding.eval( instruction_builder.build_for_cl4py_filename_inference() ))
        results = parse_idyom_results(filename)

        return results