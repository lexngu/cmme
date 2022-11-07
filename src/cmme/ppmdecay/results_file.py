import csv
from abc import ABC

from ppmdecay.model import ModelType
from ppmdecay.util import str_to_list


class ResultsFileData(ABC):
    def __init__(self, symbols, model_orders, information_contents, entropies, distributions):
        self.symbols = symbols
        self.model_orders = model_orders
        self.information_contents = information_contents
        self.entropies = entropies
        self.distributions = distributions


class PPMSimpleResultsFileData(ResultsFileData):
    def __init__(self, symbols, model_orders, information_contents, entropies, distributions):
        super().__init__(symbols, model_orders, information_contents, entropies, distributions)


class PPMDecayResultsFileData(ResultsFileData):
    def __init__(self, symbols, model_orders, information_contents, entropies, distributions, positions, times):
        super().__init__(symbols, model_orders, information_contents, entropies, distributions)
        self._positions = positions
        self._times = times


class ResultsFile:
    def __init__(self, model_type: ModelType, alphabet_levels, instructions_file_path, results_file_data_path):
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
                model_orders.append(model_order)
                information_contents.append(information_content)
                entropies.append(entropy)
                distributions.append(distribution)

        return PPMSimpleResultsFileData(symbols, model_orders, information_contents, entropies, distributions)

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
                positions.append(pos)
                times.append(time)
                model_orders.append(model_order)
                information_contents.append(information_content)
                entropies.append(entropy)
                distributions.append(distribution)

        return PPMDecayResultsFileData(symbols, model_orders, information_contents, entropies, distributions, positions, times)


