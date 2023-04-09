import tempfile
from enum import Enum
from pathlib import Path
import os
from typing import List
import re
import cl4py

from cmme.idyom.results_file import IDYOMResultsFile, parse_idyom_results


def path_with_trailing_slash(path) -> Path:
    if not isinstance(path, Path):
        path = Path(path)
    return os.path.join(path, '')

class Dataset:
    def __init__(self, id, description):
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

class LinkedViewpoint(Viewpoint):
    def __init__(self, components: List[Viewpoint]):
        """
        A linked viewpoint consists of multiple viewpoints, e.g. [CPITCH, ONSET]
        :param components:
        """
        self.components = components

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

class IDYOMInstructionBuilder:
    def __init__(self):
        self._dataset: Dataset = None
        self._target_viewpoints: List[Viewpoint] = []
        self._source_viewpoints: List[Viewpoint] = []
        self._model: IDYOMModelValue = None

        self.stm_options()
        self.ltm_options()
        self.training_options()
        self.automatically_select_source_viewpoints()
        self.output_options()
        self.caching_options()


    def dataset(self, dataset: Dataset):
        self._dataset = dataset
        return self

    def target_viewpoints(self, target_viewpoints: List[Viewpoint]):
        self._target_viewpoints = target_viewpoints
        return self

    def source_viewpoints(self, source_viewpoints: List[Viewpoint]):
        """
        Sets the source viewpoints. If you want to use IDyOM's automatic selection algorithm, ignore this function. Instead use #automatically_select_source_viewpoints(...)
        :param source_viewpoints:
        :return:
        """
        self._source_viewpoints = source_viewpoints
        return self

    def model(self, model: IDYOMModelValue):
        self._model = model
        return self

    def stm_options(self, order_bound=None, mixtures=True, update_exclusion=True, escape=IDYOMEscape.X): # original default values
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
        if pretraining_dataset_ids == None:
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

    def build_for_cl4py(self) -> tuple:
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
            source_viewpoint_names = list(map(lambda v: v.value, self._source_viewpoints))
            result.append(("'(" + " ".join(source_viewpoint_names) + ')'))
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

class IDYOMBinding:
    def __init__(self, idyom_root_path, idyom_sqlite_database_path):
        self.lisp = cl4py.Lisp(quicklisp=True)
        self.idyom_root_path = path_with_trailing_slash(idyom_root_path)
        self.idyom_sqlite_database_path: Path = path_with_trailing_slash(idyom_sqlite_database_path)

        self._setup_lisp()

    def _lisp_eval(self, cmd):
        print("> " + str(cmd))
        result = self.lisp.eval(cmd)
        print(str(result))
        return result

    def _setup_lisp(self):
        self._lisp_eval( ('defvar', 'common-lisp-user::*idyom-root*', '"'+self.idyom_root_path+'"') )
        self._lisp_eval( ('ql:quickload', '"idyom"') )
        self._lisp_eval( ('clsql:connect', ('list', '"'+self.idyom_sqlite_database_path+'"'), ':if-exists', ':old',
                   ':database-type', ':sqlite3') )

    def all_datasets(self) -> List[Dataset]:
        """
        Calls (idyom-db:describe-database)
        :return: A list of Dataset objects
        """
        result = list()

        self._lisp_eval( ('idyom-db:describe-database', ) )
        last_msg = self.lisp.msg # Requires the patched version of cl4py

        for line in last_msg.split("\n"):
            id, description = re.split("\s+", line, maxsplit=1)
            result.append(Dataset(id=id, description=description))

        return result

    def next_free_dataset_id(self) -> int:
        result = self._lisp_eval( ('idyom-db:get-next-free-id',) )
        return int(result)

    def import_midi(self, midi_files_directory_path: str, description: str, dataset_id: int) -> Dataset:
        """
        Calls (idyom-db:import-data :mid <midi_file_directory_path> <description> <dataset_id>)

        :param midi_files_directory_path: Path to directory containing midi files to import
        :param description: A string as description of the dataset
        :param dataset_id: Target dataset id. Note that if there already is a dataset with the provided id, IDyOM will fail.
        :return:
        """
        self._lisp_eval(('idyom-db:import-data', ':mid', '"' + path_with_trailing_slash(midi_files_directory_path) + '"', '"'+description+'"', dataset_id))
        dataset_id = int(re.findall("Inserting data into database: dataset (\d+)", self.lisp.msg)[0])

        return Dataset(id=dataset_id, description=description)

    def import_kern(self, krn_files_directory_path: str, description: str, dataset_id: int) -> Dataset:
        """
        Calls (idyom-db:import-data :mid <krn_files_directory_path> <description> <dataset_id>)

        :param krn_files_directory_path: Path to directory containing **kern files to import
        :param description: A string as description of the dataset
        :param dataset_id: Target dataset id. Note that if there already is a dataset with the provided id, IDyOM will fail.
        :return:
        """
        self._lisp_eval(('idyom-db:import-data', ':krn', '"'+path_with_trailing_slash(krn_files_directory_path)+'"', '"'+description+'"', dataset_id))
        dataset_id = int(re.findall("Inserting data into database: dataset (\d+)", self.lisp.msg)[0])

        return Dataset(id=dataset_id, description=description)

    def derive_viewpoint_sequence(self, dataset_id: int, composition_id: int, viewpoint_spec: List[Viewpoint]) -> List[str]:
        pass

    def run_idyom(self, instruction_builder: IDYOMInstructionBuilder) -> IDYOMResultsFile:
        result = self._lisp_eval( instruction_builder.build_for_cl4py() )

        return parse_idyom_results(self.infer_output_file_path(instruction_builder))

    def infer_output_file_path(self, instruction_builder: IDYOMInstructionBuilder) -> Path:
        filename = self._lisp_eval(instruction_builder.build_for_cl4py_filename_inference())

        return os.path.join(instruction_builder._output_options["output_path"], filename)


class IDYOMModel:
    def __init__(self):
        self.idyom_binding = IDYOMBinding("/Users/alexander/idyom", "/Users/alexander/idyom/db/database.sqlite")

    def run(self, instruction_builder: IDYOMInstructionBuilder) -> IDYOMResultsFile:
        return self.idyom_binding.run_idyom(instruction_builder)