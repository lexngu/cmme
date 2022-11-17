import csv
from abc import ABC
from pathlib import Path

from .model import ModelType
from .util.util import str_to_list


class ResultsFileData(ABC):
    def __init__(self, results_file_data_path, symbols, model_orders, information_contents, entropies, distributions):
        self.results_file_data_path = results_file_data_path
        self.symbols = symbols
        self.model_orders = model_orders
        self.information_contents = information_contents
        self.entropies = entropies
        self.distributions = distributions


class PPMSimpleResultsFileData(ResultsFileData):
    def __init__(self, results_file_data_path, symbols, model_orders, information_contents, entropies, distributions):
        super().__init__(results_file_data_path, symbols, model_orders, information_contents, entropies, distributions)


class PPMDecayResultsFileData(ResultsFileData):
    def __init__(self, results_file_data_path, symbols, model_orders, information_contents, entropies, distributions, positions, times):
        super().__init__(results_file_data_path, symbols, model_orders, information_contents, entropies, distributions)
        self._positions = positions
        self._times = times


class ResultsMetaFile:
    def __init__(self, results_file_meta_path, model_type: ModelType, alphabet_levels, instructions_file_path, results_file_data_path):
        self.results_file_meta_path = results_file_meta_path
        self._model_type = model_type
        self._alphabet_levels = alphabet_levels
        self._instructions_file_path = instructions_file_path
        self._results_file_data_path = results_file_data_path
        if model_type == ModelType.SIMPLE:
            self.results_file_data = self._parse_ppm_simple_results_file_data()
        else:
            self.results_file_data = self._parse_ppm_decay_results_file_data()

    def _parse_ppm_simple_results_file_data(self):
        symbols = []
        model_orders = []
        information_contents = []
        entropies = []
        distributions = []

        with open(self._results_file_data_path, 'r') as f:
            csvreader = csv.DictReader(f)
            for row in csvreader:
                symbol = row["symbol"]
                model_order = row["model_order"]
                information_content = row["information_content"]
                entropy = row["entropy"]
                distribution = str_to_list(row["distribution"])

                symbols.append(symbol)
                model_orders.append(int(model_order))
                information_contents.append(float(information_content))
                entropies.append(float(entropy))
                distributions.append(list(map(float, distribution)))

        return PPMSimpleResultsFileData(self._results_file_data_path, symbols, model_orders, information_contents, entropies, distributions)

    def _parse_ppm_decay_results_file_data(self):
        symbols = []
        positions = []
        times = []
        model_orders = []
        information_contents = []
        entropies = []
        distributions = []

        with open(self._results_file_data_path, 'r') as f:
            csvreader = csv.DictReader(f)
            for row in csvreader:
                symbol = row["symbol"]
                pos = row["pos"]
                time = row["time"]
                model_order = row["model_order"]
                information_content = row["information_content"]
                entropy = row["entropy"]
                distribution = str_to_list(row["distribution"])

                symbols.append(symbol)
                positions.append(int(pos))
                times.append(float(time))
                model_orders.append(int(model_order))
                information_contents.append(float(information_content))
                entropies.append(float(entropy))
                distributions.append(list(map(float, distribution)))

        return PPMDecayResultsFileData(self._results_file_data_path, symbols, model_orders, information_contents, entropies, distributions, positions, times)

def parse_results_meta_file(results_file_meta_path : Path):
    with open(results_file_meta_path, 'r') as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            model_type = ModelType(row["model_type"])
            alphabet_levels = str_to_list(row["alphabet_levels"])
            instructions_file_path = row["instructions_file_path"]
            results_file_data_path = row["results_file_data_path"]

    return ResultsMetaFile(results_file_meta_path, model_type, alphabet_levels, instructions_file_path, results_file_data_path)