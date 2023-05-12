from typing import List

from cmme.config import Config
from cmme.drex.base import DistributionType, UnprocessedPrior
from cmme.drex.binding import ResultsFile
from cmme.drex.model import DREXInstructionBuilder, DREXModel
from cmme.idyom.base import Dataset, BasicViewpoint, IDYOMResultsFile
from cmme.idyom.binding import IDYOMBinding
from cmme.idyom.model import IDYOMInstructionBuilder, IDYOMModel
from cmme.ppmdecay.base import ModelType
from cmme.ppmdecay.binding import ResultsMetaFile
from cmme.ppmdecay.model import PPMInstance, PPMModel, PPMSimpleInstance, PPMDecayInstance
from cmme.util import flatten_list


def midi_to_equal_tempered_fundamental_frequency(midi_note_number: int, standard_concert_A_pitch: int = 440, precision_dp: int = 2) -> float:
    return round(2**((midi_note_number-69)/12)*standard_concert_A_pitch, precision_dp)

class CMMETestAndPretrainingDataContainer:
    def __init__(self, target_dataset: Dataset, target_viewpoint: BasicViewpoint, pretraining_datasets: List[Dataset], idyom_binding: IDYOMBinding):
        self.target_dataset = target_dataset
        self.target_viewpoint = target_viewpoint
        self.pretraining_datasets = pretraining_datasets

        self.idyom_binding = idyom_binding

    def _dataset_as_target_viewpoint_sequence(self, dataset: Dataset) -> List[List[float]]:
        result = list()
        compositions = self.idyom_binding.all_compositions(dataset)
        for composition in compositions:
            result.append(self.idyom_binding.derive_viewpoint_sequence(composition, self.target_viewpoint))
        return result

    def _dataset_as_fundamental_frequency_sequence(self, dataset: Dataset) -> List[List[float]]:
        result = list()
        compositions = self.idyom_binding.all_compositions(dataset)
        for composition in compositions:
            sequence = self.idyom_binding.derive_viewpoint_sequence(composition, BasicViewpoint.CPITCH)
            result.append(list(map(midi_to_equal_tempered_fundamental_frequency, sequence)))

        return result

    def pretraining_datasets_as_target_viewpoint_sequence(self) -> List[List[float]]:
        result = []
        if self.pretraining_datasets is None:
            return result
        for dataset in self.pretraining_datasets:
            result.append(self._dataset_as_target_viewpoint_sequence(dataset))
        return result

    def pretraining_datasets_as_fundamental_frequency_sequence(self) -> List[List[float]]:
        result = []
        if self.pretraining_datasets is None:
            return result
        for dataset in self.pretraining_datasets:
            result.append(self._dataset_as_fundamental_frequency_sequence(dataset))
        return result

    def pretraining_datasets_alphabet(self) -> List[float]:
        return self.idyom_binding.get_alphabet(self.pretraining_datasets, self.target_viewpoint)

    def target_dataset_alphabet(self) -> List[float]:
        return self.idyom_binding.get_alphabet(self.target_dataset, self.target_viewpoint)

    def target_dataset_as_target_viewpoint_sequence(self) -> List[List[float]]:
        return self._dataset_as_target_viewpoint_sequence(self.target_dataset)

    def target_dataset_as_fundamental_frequency_sequence(self) -> List[List[float]]:
        return self._dataset_as_fundamental_frequency_sequence(self.target_dataset)


class CMMEResultsContainer:
    def __init__(self, idyom_results_file: IDYOMResultsFile = None, ppmdecay_results_meta_file: ResultsMetaFile = None, drex_results_file: ResultsFile = None):
        self.idyom_results_file = idyom_results_file
        self.ppmdecay_results_meta_file = ppmdecay_results_meta_file
        self.drex_results_file = drex_results_file

class CMME:
    def __init__(self):
        self._idyom_instruction_builder = IDYOMInstructionBuilder()
        self._ppm_instruction_builder: PPMInstance = None
        self._drex_instruction_builder = DREXInstructionBuilder()

        self._idyom_runner: IDYOMModel = None
        self._ppmdecay_runner: PPMModel = None
        self._drex_runner: DREXModel = None

    def idyom(self) -> IDYOMInstructionBuilder:
        return self._idyom_instruction_builder

    def ppmdecay(self, model_type: ModelType) -> PPMInstance:
        if model_type == ModelType.SIMPLE:
            self._ppm_instruction_builder = PPMSimpleInstance()
        elif model_type == ModelType.DECAY:
            self._ppm_instruction_builder = PPMDecayInstance()

        return self._ppm_instruction_builder

    def drex(self) -> DREXInstructionBuilder:
        return self._drex_instruction_builder

    def run(self, dc: CMMETestAndPretrainingDataContainer) -> CMMEResultsContainer:
        if dc.target_viewpoint != BasicViewpoint.CPITCH:
            raise ValueError("At the moment, only target_viewpoint == BasicViewpoint.CPITCH is supported!")

        self._init_runners()

        self._idyom_instruction_builder\
            .dataset(dc.target_dataset)\
            .target_viewpoints([dc.target_viewpoint])\
            .source_viewpoints([dc.target_viewpoint])\
            .training_options(dc.pretraining_datasets)

        alphabet_levels = dc.pretraining_datasets_alphabet() if dc.pretraining_datasets != None else dc.target_dataset_alphabet()

        self._ppm_instruction_builder.input_sequence(flatten_list(dc.target_dataset_as_target_viewpoint_sequence()))\
            .alphabet_levels(alphabet_levels)
        # TODO PPM without pre-training so far...

        drex_input_sequence = flatten_list(dc.target_dataset_as_fundamental_frequency_sequence())
        self._drex_instruction_builder.input_sequence(drex_input_sequence)
        prior_distribution_type = DistributionType.GAUSSIAN
        if dc.pretraining_datasets is None or (isinstance(dc.pretraining_datasets, list) and len(dc.pretraining_datasets) == 0):
            prior_input_sequence = drex_input_sequence
            print("Warning! Prior's input sequence was set to input sequence due to lack of additional pretraining data.")
        else:
            prior_input_sequence = flatten_list(dc.pretraining_datasets_as_fundamental_frequency_sequence())
        prior_D = 4 # TODO remove hard coded value

        self._drex_instruction_builder.prior(UnprocessedPrior(prior_distribution_type, prior_input_sequence, prior_D))

        print("Run IDyOM...")
        idyom_result = self._idyom_runner.run(self._idyom_instruction_builder)
        print("Run PPM...")
        ppmdecay_result = self._ppmdecay_runner.run()
        print("Run DREX...")
        drex_result = self._drex_runner.run()

        return CMMEResultsContainer(idyom_result, ppmdecay_result, drex_result)

    def _init_runners(self):
        if self._idyom_runner is None:
            self._idyom_runner = IDYOMModel(Config().idyom_root_path(), Config().idyom_database_path())
        if self._ppmdecay_runner is None:
            self._ppmdecay_runner = PPMModel(self._ppm_instruction_builder)
        if self._drex_runner is None:
            self._drex_runner = DREXModel(self._drex_instruction_builder)