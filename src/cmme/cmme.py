from typing import List, Union

from cmme.config import Config
from cmme.drex.binding import DREXResultsFile
from cmme.drex.model import DREXInstructionBuilder
from cmme.drex.worker import DREXModel
from cmme.idyom.base import Dataset, BasicViewpoint
from cmme.idyom.binding import IDYOMResultsFile
from cmme.idyom.idyom_database import IdyomDatabase
from cmme.idyom.model import IDYOMInstructionBuilder, IDYOMModel
from cmme.ppmdecay.base import ModelType
from cmme.ppmdecay.binding import PPMResultsMetaFile
from cmme.ppmdecay.model import PPMInstance, PPMModel, PPMSimpleInstance, PPMDecayInstance

freq_to_midi = {
    8.18: 0, 8.66: 1, 9.18: 2, 9.72: 3, 10.3: 4, 10.91: 5, 11.56: 6, 12.25: 7, 12.98: 8, 13.75: 9, 14.57: 10, 15.43: 11, 16.35: 12, 17.32: 13, 18.35: 14, 19.45: 15, 20.6: 16, 21.83: 17, 23.12: 18, 24.5: 19, 25.96: 20, 27.5: 21, 29.14: 22, 30.87: 23, 32.7: 24, 34.65: 25, 36.71: 26, 38.89: 27, 41.2: 28, 43.65: 29, 46.25: 30, 49.0: 31, 51.91: 32, 55.0: 33, 58.27: 34, 61.74: 35, 65.41: 36, 69.3: 37, 73.42: 38, 77.78: 39, 82.41: 40, 87.31: 41, 92.5: 42, 98.0: 43, 103.83: 44, 110.0: 45, 116.54: 46, 123.47: 47, 130.81: 48, 138.59: 49, 146.83: 50, 155.56: 51, 164.81: 52, 174.61: 53, 185.0: 54, 196.0: 55, 207.65: 56, 220.0: 57, 233.08: 58, 246.94: 59, 261.63: 60, 277.18: 61, 293.66: 62, 311.13: 63, 329.63: 64, 349.23: 65, 369.99: 66, 392.0: 67, 415.3: 68, 440.0: 69, 466.16: 70, 493.88: 71, 523.25: 72, 554.37: 73, 587.33: 74, 622.25: 75, 659.26: 76, 698.46: 77, 739.99: 78, 783.99: 79, 830.61: 80, 880.0: 81, 932.33: 82, 987.77: 83, 1046.5: 84, 1108.73: 85, 1174.66: 86, 1244.51: 87, 1318.51: 88, 1396.91: 89, 1479.98: 90, 1567.98: 91, 1661.22: 92, 1760.0: 93, 1864.66: 94, 1975.53: 95, 2093.0: 96, 2217.46: 97, 2349.32: 98, 2489.02: 99, 2637.02: 100, 2793.83: 101, 2959.96: 102, 3135.96: 103, 3322.44: 104, 3520.0: 105, 3729.31: 106, 3951.07: 107, 4186.01: 108, 4434.92: 109, 4698.64: 110, 4978.03: 111, 5274.04: 112, 5587.65: 113, 5919.91: 114, 6271.93: 115, 6644.88: 116, 7040.0: 117, 7458.62: 118, 7902.13: 119, 8372.02: 120, 8869.84: 121, 9397.27: 122, 9956.06: 123, 10548.08: 124, 11175.3: 125, 11839.82: 126, 12543.85: 127
}


def midi_to_equal_tempered_fundamental_frequency(midi_note_number: int, standard_concert_A_pitch: int = 440, precision_dp: int = 2) -> float:
    return round(2**((midi_note_number-69)/12)*standard_concert_A_pitch, precision_dp)


def equal_tempered_fundamental_frequency_to_midi(fundamental_frequency: float) -> int:
    freq = round(fundamental_frequency, 2)
    if freq in freq_to_midi:
        return freq_to_midi[freq]
    else:
        raise ValueError("Invalid frequency! Could not convert to midi note number.")


def cmme_idyom_database() -> IdyomDatabase:
    """
    Return preconfigured IdyomDatabase object, using the values of the CMME config file.

    Returns
    -------
    IdyomDatabase
        IdyomDatabase
    """
    return IdyomDatabase(Config().idyom_root_path(), Config().idyom_database_path())


class CMMETestAndPretrainingDataContainer:
    def __init__(self, target_dataset: Dataset, target_viewpoint: BasicViewpoint, pretraining_datasets: List[Dataset], idyom_database: IdyomDatabase):
        self.target_dataset = target_dataset
        self.target_viewpoint = target_viewpoint
        self.pretraining_datasets = pretraining_datasets

        self.idyom_database = idyom_database

    def _dataset_as_target_viewpoint_sequence(self, dataset: Union[int, Dataset]) -> List[List[float]]:
        result = list()
        compositions = self.idyom_database.get_all_compositions(dataset)
        for composition in compositions:
            sequence = self.idyom_database.encode_composition(composition, self.target_viewpoint)
            if len(sequence) > 0:
                result.append(sequence)
        return result

    def _dataset_as_fundamental_frequency_sequence(self, dataset: Union[int, Dataset]) -> List[List[float]]:
        result = list()
        compositions = self.idyom_database.get_all_compositions(dataset)
        for composition in compositions:
            sequence = self.idyom_database.encode_composition(composition, BasicViewpoint.CPITCH)
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
        return self.idyom_database.get_dataset_alphabet(self.pretraining_datasets, self.target_viewpoint)

    def target_dataset_alphabet(self) -> List[float]:
        return self.idyom_database.get_dataset_alphabet(self.target_dataset, self.target_viewpoint)

    def target_dataset_as_target_viewpoint_sequence(self) -> List[List[float]]:
        return self._dataset_as_target_viewpoint_sequence(self.target_dataset)

    def target_dataset_as_fundamental_frequency_sequence(self) -> List[List[float]]:
        return self._dataset_as_fundamental_frequency_sequence(self.target_dataset)


class CMMEResultsContainer:
    def __init__(self, idyom_results_file: IDYOMResultsFile = None, ppmdecay_results_meta_file: PPMResultsMetaFile = None, drex_results_file: DREXResultsFile = None):
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