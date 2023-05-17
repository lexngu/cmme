from typing import List

from cmme.config import Config
from cmme.drex.binding import ResultsFile
from cmme.drex.model import DREXInstructionBuilder, DREXModel
from cmme.idyom.base import Dataset, BasicViewpoint, IDYOMResultsFile
from cmme.idyom.binding import IDYOMBinding
from cmme.idyom.model import IDYOMInstructionBuilder, IDYOMModel
from cmme.ppmdecay.base import ModelType
from cmme.ppmdecay.binding import ResultsMetaFile
from cmme.ppmdecay.model import PPMInstance, PPMModel, PPMSimpleInstance, PPMDecayInstance


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
            sequence = self.idyom_binding.derive_viewpoint_sequence(composition, self.target_viewpoint)
            if len(sequence) > 0:
                result.append(sequence)
        return result

    def _dataset_as_fundamental_frequency_sequence(self, dataset: Dataset) -> List[List[float]]:
        result = list()
        compositions = self.idyom_binding.all_compositions(dataset)
        for composition in compositions:
            sequence = self.idyom_binding.derive_viewpoint_sequence(composition, BasicViewpoint.CPITCH)
            if len(sequence) > 0:
                result.append(list(map(midi_to_equal_tempered_fundamental_frequency, sequence)))

        return result

    def pretraining_datasets_as_target_viewpoint_sequence(self) -> List[List[float]]:
        result = []
        if self.pretraining_datasets is None:
            return result
        for dataset in self.pretraining_datasets:
            result[len(result):] = self._dataset_as_target_viewpoint_sequence(dataset)
        return result

    def pretraining_datasets_as_fundamental_frequency_sequence(self) -> List[List[float]]:
        result = []
        if self.pretraining_datasets is None:
            return result
        for dataset in self.pretraining_datasets:
            result[len(result):] = self._dataset_as_fundamental_frequency_sequence(dataset)
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