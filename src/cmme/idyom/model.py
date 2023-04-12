import tempfile
import os
from .base import *
from .binding import IDYOMBinding
from cmme.util import flatten_list


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


    def dataset(self, dataset: Dataset):
        if not isinstance(dataset, Dataset):
            raise ValueError("dataset must be an instance of Dataset!")
        self._dataset = dataset
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

    def stm_options(self, order_bound: int =None, mixtures=True, update_exclusion=True, escape=IDYOMEscape.X): # original default values
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

    def training_options(self, pretraining_dataset_ids = None, resampling_folds_count = 10, exclusively_to_be_used_resampling_fold_indices = None):
        """
        Training options affect all model configurations (STM, LTM, LTM+, BOTH, BOTH+).
        If STM, there is no training, however, the pretraining datasets provide the "viewpoint domain" (i.e. PPM's "alphabet") for the STM model.
        For all the remaining configurations (LTM, ...), this pre-training happens before IDyOM's resampling procedure.
        :param pretraining_dataset_ids: If None, training options are reset.
        :param resampling_folds_count:
        :param exclusively_to_be_used_resampling_fold_indices: If None, IDyOM will use all folds
        :return:
        """
        if pretraining_dataset_ids is None:
            self._training_options = {}
        else:
            self._training_options = {
                "pretraining_dataset_ids": pretraining_dataset_ids,
                "resampling_folds_count": resampling_folds_count,
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
        if output_path == None:
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

        if self._dataset == None:
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

    def build_for_cl4py(self) -> tuple:
        self.assert_is_valid()
        result = list()

        # (idyom:idyom ...)
        result.append("idyom:idyom")
        # (idyom:idyom <dataset-id> ...)
        result.append(self._dataset.id)
        # (idyom:idyom <dataset-id> <target-viewpoints> ...)
        target_viewpoint_names = list(map(lambda v: v.value, self._target_viewpoints))
        result.append(('quote', '('+" ".join(target_viewpoint_names)+')'))
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> ...)
        if not self._select_options == {}: # if not empty
            result.append(":select")
        else:
            source_viewpoint_names = viewpoints_list_as_lisp_string(self._source_viewpoints)
            result.append(('quote', source_viewpoint_names))
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> ...)
        result.append(":models")
        result.append(self._model.value)
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] ...)
        if self._stm_options:
            result.append(":stmo")
            order_bound = self._stm_options["order_bound"] if self._stm_options["order_bound"] else "nil"
            mixtures = "t" if self._stm_options["mixtures"] else "nil"
            update_exclusion = "t" if self._stm_options["update_exclusion"] else "nil"
            escape = self._stm_options["escape"].value
            result.append(("quote", "(:order-bound "+order_bound+" :mixtures "+mixtures+" :update-exclusion "+update_exclusion+" :escape "+escape+")"))
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...] ...)
        if self._ltm_options:
            result.append(":ltmo")
            order_bound = str(self._stm_options["order_bound"]) if self._stm_options["order_bound"] else "nil"
            mixtures = "t" if self._stm_options["mixtures"] else "nil"
            update_exclusion = "t" if self._stm_options["update_exclusion"] else "nil"
            escape = self._stm_options["escape"].value
            result.append(("quote", "(:order-bound " + order_bound + " :mixtures " + mixtures + " :update-exclusion " + update_exclusion + " :escape " + escape + ")"))
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] ...)
        if self._training_options:
            pretraining_ids = list(map(str, self._training_options["pretraining_dataset_ids"]))
            result.append(":pretraining-ids")
            result.append(("quote", "("+" ".join(pretraining_ids)+")"))

            result.append(":k")
            result.append(self._training_options["k"])

            if self._training_options["exclusively_to_be_used_resampling_fold_indices"]:
                result.append(":resampling-indices")
                indices = list(map(str, self._training_options["exclusively_to_be_used_resampling_fold_indices"]))
                result.append(("quote", " ".join(indices)))
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] [:basis ... :dp ... :max-links ... :min-links ... :viewpoint-selection-output ...] ...)
        if self._select_options:
            result.append(":basis")
            basis = self._select_options["basis"]
            if isinstance(basis, IDYOMViewpointSelectionBasis):
                result.append(basis.value)
            elif isinstance(basis, list):
                result.append(("quote", "("+" ".join(basis)+")"))
            else:
                print("ERROR")
            result.append(":dp")
            dp = str(self._select_options["dp"]) if self._select_options["dp"] else "nil"
            result.append(dp)
            result.append(":max-links")
            result.append(self._select_options["max_links"])
            result.append(":min-links")
            result.append(self._select_options["min_links"])
            result.append(":viewpoint-selection-output")
            result.append('"'+self._select_options["viewpoint_selection_output"]+'"')
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] [:basis ... :dp ... :max-links ... :min-links ... :viewpoint-selection-output ...]
        # [:output-path ... :detail ... :overwrite ... :separator ...] ...)
        result.append(":output-path")
        result.append('"'+self._output_options["output_path"]+'"' if self._output_options["output_path"] else "nil")
        result.append(":detail")
        result.append(self._output_options["detail"])
        result.append(":overwrite")
        result.append("t" if self._output_options["overwrite"] else "nil")
        result.append(":separator")
        result.append('"'+self._output_options["separator"]+'"')
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] [:basis ... :dp ... :max-links ... :min-links ... :viewpoint-selection-output ...]
        # [:output-path ... :detail ... :overwrite ... :separator ...] [:use-resampling-set-cache? ... :use-ltms-cache? ...])
        result.append(":use-resampling-set-cache?")
        result.append("t" if self._caching_options["use_resampling_set_cache"] else "nil")
        result.append(":use-ltms-cache?")
        result.append("t" if self._caching_options["use_ltms_cache"] else "nil")

        return tuple(result)

    def build_for_cl4py_filename_inference(self) -> tuple:
        self.assert_is_valid()
        result = list()

        # (apps:dataset-modelling-filename ...)
        result.append("apps:dataset-modelling-filename")
        # (idyom:idyom <dataset-id> ...)
        result.append(self._dataset.id)
        # (idyom:idyom <dataset-id> <target-viewpoints> ...)
        target_viewpoint_names = list(map(lambda v: v.value, self._target_viewpoints))
        result.append(('quote', '(' + " ".join(target_viewpoint_names) + ')'))
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> ...)
        if not self._select_options == {}:  # if not empty
            result.append(":select")
        else:
            source_viewpoint_names = list(map(lambda v: v.value, self._source_viewpoints))
            result.append(('quote', '(' + " ".join(source_viewpoint_names) + ')'))
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> ...)
        result.append(":models")
        result.append(self._model.value)
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] ...)
        if self._stm_options:
            result.append(":stmo")
            order_bound = self._stm_options["order_bound"] if self._stm_options["order_bound"] else "nil"
            mixtures = "t" if self._stm_options["mixtures"] else "nil"
            update_exclusion = "t" if self._stm_options["update_exclusion"] else "nil"
            escape = self._stm_options["escape"].value
            result.append(("quote",
                           "(:order-bound " + order_bound + " :mixtures " + mixtures + " :update-exclusion " + update_exclusion + " :escape " + escape + ")"))
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...] ...)
        if self._ltm_options:
            result.append(":ltmo")
            order_bound = str(self._stm_options["order_bound"]) if self._stm_options["order_bound"] else "nil"
            mixtures = "t" if self._stm_options["mixtures"] else "nil"
            update_exclusion = "t" if self._stm_options["update_exclusion"] else "nil"
            escape = self._stm_options["escape"].value
            result.append(("quote",
                           "(:order-bound " + order_bound + " :mixtures " + mixtures + " :update-exclusion " + update_exclusion + " :escape " + escape + ")"))
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] ...)
        if self._training_options:
            pretraining_ids = list(map(str, self._training_options["pretraining_dataset_ids"]))
            result.append(":pretraining-ids")
            result.append(("quote", "(" + " ".join(pretraining_ids) + ")"))

            result.append(":k")
            result.append(self._training_options["k"])

            if self._training_options["exclusively_to_be_used_resampling_fold_indices"]:
                result.append(":resampling-indices")
                indices = list(map(str, self._training_options["exclusively_to_be_used_resampling_fold_indices"]))
                result.append(("quote", " ".join(indices)))
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] [:basis ... :dp ... :max-links ... :min-links ... :viewpoint-selection-output ...] ...)
        if self._select_options:
            result.append(":basis")
            basis = self._select_options["basis"]
            if isinstance(basis, IDYOMViewpointSelectionBasis):
                result.append(basis.value)
            elif isinstance(basis, list):
                result.append(("quote", "(" + " ".join(basis) + ")"))
            else:
                print("ERROR")
            result.append(":dp")
            dp = str(self._select_options["dp"]) if self._select_options["dp"] else "nil"
            result.append(dp)
            result.append(":max-links")
            result.append(self._select_options["max_links"])
            result.append(":min-links")
            result.append(self._select_options["min_links"])
            result.append(":viewpoint-selection-output")
            result.append('"' + self._select_options["viewpoint_selection_output"] + '"')
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] [:basis ... :dp ... :max-links ... :min-links ... :viewpoint-selection-output ...]
        # [:output-path ... :detail ... :overwrite ... :separator ...] ...)
        result.append(":detail")
        result.append(self._output_options["detail"])

        # extension
        result.append(":extension")
        result.append('".dat"')

        return tuple(result)

    def build_for_lisp(self) -> str:
        self.assert_is_valid()
        result = list()

        # (idyom:idyom ...)
        result.append("(idyom:idyom")
        # (idyom:idyom <dataset-id> ...)
        result.append(str(self._dataset.id))
        # (idyom:idyom <dataset-id> <target-viewpoints> ...)
        target_viewpoint_names = list(map(lambda v: v.value, self._target_viewpoints))
        result.append(("'(" + " ".join(target_viewpoint_names) + ')'))
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> ...)
        if not self._select_options == {}:  # if not empty
            result.append(":select")
        else:
            source_viewpoint_names = viewpoints_list_as_lisp_string(self._source_viewpoints)
            result.append("'" + source_viewpoint_names)
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> ...)
        result.append(":models")
        result.append(self._model.value)
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] ...)
        if self._stm_options:
            result.append(":stmo")
            order_bound = str(self._stm_options["order_bound"]) if self._stm_options["order_bound"] else "nil"
            mixtures = "t" if self._stm_options["mixtures"] else "nil"
            update_exclusion = "t" if self._stm_options["update_exclusion"] else "nil"
            escape = self._stm_options["escape"].value
            result.append("'(:order-bound " + order_bound + " :mixtures " + mixtures + " :update-exclusion " + update_exclusion + " :escape " + escape + ")")
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...] ...)
        if self._ltm_options:
            result.append(":ltmo")
            order_bound = str(self._stm_options["order_bound"]) if self._stm_options["order_bound"] else "nil"
            mixtures = "t" if self._stm_options["mixtures"] else "nil"
            update_exclusion = "t" if self._stm_options["update_exclusion"] else "nil"
            escape = self._stm_options["escape"].value
            result.append(("'(:order-bound " + order_bound + " :mixtures " + mixtures + " :update-exclusion " + update_exclusion + " :escape " + escape + ")"))
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] ...)
        if self._training_options:
            pretraining_ids = list(map(str, self._training_options["pretraining_dataset_ids"]))
            result.append(":pretraining-ids")
            result.append(("quote", "(" + " ".join(pretraining_ids) + ")"))

            result.append(":k")
            result.append(self._training_options["k"])

            if self._training_options["exclusively_to_be_used_resampling_fold_indices"]:
                result.append(":resampling-indices")
                indices = list(map(str, self._training_options["exclusively_to_be_used_resampling_fold_indices"]))
                result.append(("quote", " ".join(indices)))
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] [:basis ... :dp ... :max-links ... :min-links ... :viewpoint-selection-output ...] ...)
        if self._select_options:
            result.append(":basis")
            basis = self._select_options["basis"]
            if isinstance(basis, IDYOMViewpointSelectionBasis):
                result.append(basis.value)
            elif isinstance(basis, list):
                result.append(("'(" + " ".join(basis) + ")"))
            else:
                print("ERROR")
            result.append(":dp")
            dp = str(self._select_options["dp"]) if self._select_options["dp"] else "nil"
            result.append(str(dp))
            result.append(":max-links")
            result.append(self._select_options["max_links"])
            result.append(":min-links")
            result.append(self._select_options["min_links"])
            result.append(":viewpoint-selection-output")
            result.append('"' + self._select_options["viewpoint_selection_output"] + '"')
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] [:basis ... :dp ... :max-links ... :min-links ... :viewpoint-selection-output ...]
        # [:output-path ... :detail ... :overwrite ... :separator ...] ...)
        result.append(":output-path")
        result.append('"' + self._output_options["output_path"] + '"' if self._output_options["output_path"] else "nil")
        result.append(":detail")
        result.append(str(self._output_options["detail"]))
        result.append(":overwrite")
        result.append("t" if self._output_options["overwrite"] else "nil")
        result.append(":separator")
        result.append('"' + self._output_options["separator"] + '"')
        # (idyom:idyom <dataset-id> <target-viewpoints> <source-viewpoints> :models <models> [:stmo ...] [:ltmo ...]
        # [:pretraining-ids ... :k ... :resampling-indices ...] [:basis ... :dp ... :max-links ... :min-links ... :viewpoint-selection-output ...]
        # [:output-path ... :detail ... :overwrite ... :separator ...] [:use-resampling-set-cache? ... :use-ltms-cache? ...])
        result.append(":use-resampling-set-cache?")
        result.append("t" if self._caching_options["use_resampling_set_cache"] else "nil")
        result.append(":use-ltms-cache?")
        result.append("t" if self._caching_options["use_ltms_cache"] else "nil")

        result.append(")")

        return " ".join(result)

class IDYOMModel:
    def __init__(self, idyom_root_path: Path, idyom_database_path: Path):
        self.idyom_binding = IDYOMBinding(str(idyom_root_path.resolve()), str(idyom_database_path.resolve()))

    def import_midi(self, midi_files_directory_path: str, description: str, dataset_id: int = None) -> Dataset:
        """

        :param midi_files_directory_path:
        :param description:
        :param dataset_id: If None, a valid value will be determined automatically
        :return:
        """
        if dataset_id is None:
            dataset_id = self.idyom_binding.next_free_dataset_id()

        return self.idyom_binding.import_midi(midi_files_directory_path, description, dataset_id)

    def import_kern(self, krn_files_directory_path: str, description: str, dataset_id: int = None) -> Dataset:
        """

        :param krn_files_directory_path:
        :param description:
        :param dataset_id: If None, a valid value will be determined automatically
        :return:
        """
        if dataset_id is None:
            dataset_id = self.idyom_binding.next_free_dataset_id()

        return self.idyom_binding.import_kern(krn_files_directory_path, description, dataset_id)

    def all_datasets(self) -> List[Dataset]:
        return self.idyom_binding.all_datasets()

    def run(self, instruction_builder: IDYOMInstructionBuilder) -> IDYOMResultsFile:
        self.idyom_binding.eval( instruction_builder.build_for_cl4py() )

        filename = os.path.join(instruction_builder._output_options["output_path"], self.idyom_binding.eval( instruction_builder.build_for_cl4py_filename_inference() ))
        results = parse_idyom_results(filename)

        return results