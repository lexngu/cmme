import re
import tempfile

from .base import *
from .binding import *
from .idyom_database import IdyomDatabase
from .util import *
from ..lib.instructions_file import InstructionsFile
from ..lib.model import ModelBuilder, Model


def viewpoint_name_to_viewpoint(name: str) -> Viewpoint:
    """
    Return the associated viewpoint object.

    Parameters
    ----------
    name
        Viewpoint name

    Returns
    -------
    Viewpoint
        Viewpoint object, if viewpoint could be determined. ValueError otherwise.
    """
    candidates = [BasicViewpoint, DerivedViewpoint, ThreadedViewpoint, TestViewpoint]
    result = None
    for candidate in candidates:
        try:
            result = candidate(name)
        except ValueError:
            pass

    if result is None:
        raise ValueError("Viewpoint with name={} invalid!".format(name))

    return result


class IDYOMInstructionBuilder(ModelBuilder):
    def to_instructions_file(self) -> IDYOMInstructionsFile:
        self.assert_is_valid()
        return IDYOMInstructionsFile(self._dataset, self._target_viewpoints, self._source_viewpoints,
                                     self._model, self._stm_options, self._ltm_options, self._training_options,
                                     self._select_options, self._output_options, self._caching_options,
                                     self._idyom_root_path, self._idyom_database_path)

    def __init__(self):
        super().__init__()
        self._caching_options = None
        self._output_options = None
        self._select_options = None
        self._training_options = None
        self._ltm_options = None
        self._stm_options = None
        self._dataset = None
        self._dataset: Dataset
        self._target_viewpoints: List[Viewpoint] = []
        self._source_viewpoints: List[Viewpoint] = []
        self._model: IDYOMModelValue = IDYOMModelValue.BOTH_PLUS

        self.stm_options()
        self.ltm_options()
        self.training_options()
        self.automatically_select_source_viewpoints()
        self.output_options()
        self.caching_options()

        self._idyom_root_path = None
        self._idyom_database_path = None

    def dataset(self, dataset_or_id):
        if isinstance(dataset_or_id, Dataset):
            self._dataset = dataset_or_id
        elif isinstance(dataset_or_id, int):
            self._dataset = Dataset(id=dataset_or_id, description="(Unchecked) id value")
        else:
            raise ValueError("Dataset invalid! Provide an id value or Dataset instance.")
        return self

    def target_viewpoints(self, target_viewpoints: Union[str, BasicViewpoint, List[BasicViewpoint]]):
        if isinstance(target_viewpoints, BasicViewpoint):
            target_viewpoints = [target_viewpoints]
        elif isinstance(target_viewpoints, str):
            target_viewpoints = [viewpoint_name_to_viewpoint(target_viewpoints)]
        elif isinstance(target_viewpoints, list):
            _tmp = []
            for e in target_viewpoints:
                if isinstance(e, BasicViewpoint):
                    _tmp.append(e)
                elif isinstance(e, str):
                    _tmp.append(viewpoint_name_to_viewpoint(e))
                else:
                    raise ValueError("target_viewpoints must not contain anything except for strings, "
                                     "or Viewpoint objects!")
            target_viewpoints = _tmp
        else:
            raise ValueError("target_viewpoints must be a string, a Viewpoint, or a list of these.")

        self._target_viewpoints = target_viewpoints

        return self

    def source_viewpoints(self, source_viewpoints: List[Viewpoint]):
        """
        Sets the source viewpoints. If you want to use IDyOM's automatic selection algorithm, ignore this function.
        Instead, use #automatically_select_source_viewpoints(...)

        :param source_viewpoints:
        :return:
        """
        if isinstance(source_viewpoints, BasicViewpoint):
            source_viewpoints = [source_viewpoints]
        elif isinstance(source_viewpoints, str):
            source_viewpoints = [viewpoint_name_to_viewpoint(source_viewpoints)]
        elif isinstance(source_viewpoints, list):
            _tmp = []
            for e in source_viewpoints:
                if isinstance(e, BasicViewpoint):
                    _tmp.append(e)
                elif isinstance(e, str):
                    _tmp.append(viewpoint_name_to_viewpoint(e))
                else:
                    raise ValueError("target_viewpoints must not contain anything except for strings, "
                                     "or Viewpoint objects!")
            source_viewpoints = _tmp
        else:
            raise ValueError("source_viewpoints must be a string, a Viewpoint, or a list of these.")

        self._source_viewpoints = source_viewpoints
        return self

    def model(self, model: IDYOMModelValue):
        self._model = model
        return self

    def stm_options(self, order_bound: int = None, mixtures=None, update_exclusion=None,
                    escape=None):
        self._stm_options = {
            "order_bound": order_bound,
            "mixtures": mixtures,
            "update_exclusion": update_exclusion,
            "escape": escape
        }
        return self

    def ltm_options(self, order_bound=None, mixtures=None, update_exclusion=None,
                    escape=None):
        self._ltm_options = {
            "order_bound": order_bound,
            "mixtures": mixtures,
            "update_exclusion": update_exclusion,
            "escape": escape
        }
        return self

    def training_options(self, pretraining_dataset_ids=None, resampling_folds_count_k=None,
                         exclusively_to_be_used_resampling_fold_indices=None):
        """
        Training options affect all model configurations (STM, LTM, LTM+, BOTH, BOTH+).
        If STM, there is no training, however, the pretraining datasets provide the "viewpoint domain"
        (i.e. PPM's "alphabet") for the STM model.
        For all the remaining configurations (LTM, ...), this pre-training happens before IDyOM's resampling procedure.
        :param pretraining_dataset_ids: If None, training options are reset.
        :param resampling_folds_count_k:
        :param exclusively_to_be_used_resampling_fold_indices: If None, IDyOM will use all folds
        :return:
        """
        if pretraining_dataset_ids is None and resampling_folds_count_k is None:
            self._training_options = {}
        else:
            self._training_options = {
                "pretraining_dataset_ids": pretraining_dataset_ids,
                "resampling_folds_count_k": resampling_folds_count_k,
                "exclusively_to_be_used_resampling_fold_indices": exclusively_to_be_used_resampling_fold_indices
            }
        return self

    def automatically_select_source_viewpoints(self, basis: IDYOMViewpointSelectionBasis = None, dp=None,
                                               max_links=None, min_links=None, viewpoint_selection_output=None):
        """

        :param basis: If None, select options are reset.
        :param dp:
        :param max_links:
        :param min_links:
        :param viewpoint_selection_output:
        :return:
        """
        if basis is None:
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

    def output_options(self, output_path: Union[str, Path] = tempfile.gettempdir(), detail=None,
                       overwrite=None, separator=None):
        if output_path is None:
            raise ValueError("None is not an allowed value for output_path!")

        self._output_options = {
            "output_path": output_path,
            "detail": detail,
            "overwrite": overwrite,
            "separator": separator
        }
        return self

    def caching_options(self, use_resampling_set_cache=None, use_ltms_cache=None):
        self._caching_options = {
            "use_resampling_set_cache": use_resampling_set_cache,
            "use_ltms_cache": use_ltms_cache
        }
        return self

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

        return is_valid, msgs

    def assert_is_valid(self):
        builder_is_valid, builder_error_msgs = self._is_valid()
        if not builder_is_valid:
            raise ValueError(" ".join(builder_error_msgs))
        return

    def idyom_root_path(self, idyom_root_path):
        self._idyom_root_path = idyom_root_path
        return self

    def idyom_database_path(self, idyom_database_path):
        self._idyom_database_path = idyom_database_path
        return self

class IDYOMModel(Model):
    @staticmethod
    def run_instructions_file_at_path(file_path: Union[str, Path]) -> IDYOMResultsFile:
        out, err = run_idyom_instructions_file(file_path)
        out_last_line = out.decode('utf-8').split("\n")[-1]
        search_results = re.search(r"results_file_path=(.+)\"", out_last_line)
        results_file_path = search_results.groups()[0] if search_results else None
        if len(err) > 0:
            raise ValueError("Error! {}".format(err))
        if results_file_path is None or not os.path.exists(results_file_path):
            raise ValueError("Could not determine results_file_path!")

        return IDYOMResultsFile.load(results_file_path)

    def __init__(self):
        super().__init__()
