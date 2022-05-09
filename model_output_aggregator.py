import pandas as pd
import scipy.io as sio


class ModelOutputAggregator:
    """This class parses models' output files and aggregates their data in a single data frame."""

    def __init__(self, ppm_output_parameters, drex_output_parameters):
        self._ppm_output_parameters  = ppm_output_parameters
        self._drex_output_parameters = drex_output_parameters

        if not self._input_sequences_match():
            print(self._ppm_output_parameters.input_sequence)
            print(self._drex_output_parameters.input_sequence)
            raise ValueError("Input sequences of each PPM and DREX do not match!")
        else:
            self.input_sequence = self._drex_output_parameters.input_sequence
        self.df = self._build_aggregate_df()

    def _input_sequences_match(self):
        # ToDo: *.csv and *.mat files have different number precision. so we cannot trivially use "=="
        ppmis  = self._ppm_output_parameters.input_sequence
        drexis = self._drex_output_parameters.input_sequence

        length_matches = len(ppmis) == len(drexis)
        sum_nearly_equal = abs(sum(ppmis) - sum(drexis)) < 1

        return length_matches and sum_nearly_equal

    def _build_aggregate_df(self):
        data_columns = ["observation_idx", "observation", 
                        "ppm_information_content", "drex_surprisal", 
                        "ppm_alphabet_size", "ppm_predictions", "drex_predictions", 
                        "ppm_model_order", "drex_context_beliefs", 
                        "ppm_entropy", "drex_entropy", 
                        "drex_cd_threshold", "drex_cd_probability", "drex_cd_changepoint", 
                        "drex_bd"]
        data = pd.DataFrame([], columns = data_columns)

        nullValue = -1
        for idx, v in enumerate(self.input_sequence):
            data.loc[idx] = [idx, v, 
                             nullValue, nullValue, 
                             nullValue, [], nullValue, 
                             nullValue, nullValue, 
                             nullValue, nullValue,
                             self._drex_output_parameters.change_decision_threshold, nullValue, self._drex_output_parameters.change_decision_changepoint, 
                             nullValue]
        data.loc[len(self.input_sequence)] = [nullValue, nullValue, 
                                              nullValue, nullValue, 
                                              nullValue, [], nullValue, 
                                              nullValue, nullValue, 
                                              nullValue, nullValue,
                                              nullValue, nullValue, nullValue, 
                                              nullValue] # add one more row due to D-REX context beliefs goes beyond input sequence
            
        for idx, row in self._ppm_output_parameters.data_frame.iterrows():
            symbol_idx = int(row["symbol_idx"])
            distribution_idx = int(row["distribution_idx"])
            #distribution_symbol = row["distribution_symbol"]
            distribution = row["distribution"]
            #symbol = row["symbol"]
            model_order = int(row["model_order"])
            information_content = row["information_content"]
            entropy = row["entropy"]
            
            if distribution_idx == 1:
                data.at[symbol_idx-1, "ppm_alphabet_size"] = self._ppm_output_parameters.alphabet_size
                data.at[symbol_idx-1, "ppm_information_content"] = information_content
                data.at[symbol_idx-1, "ppm_model_order"] = model_order
                data.at[symbol_idx-1, "ppm_entropy"] = entropy
                data.at[symbol_idx-1, "ppm_predictions"] = []    
            predictions = data.at[symbol_idx-1, "ppm_predictions"]
            predictions.append(distribution)

        for idx, row in enumerate(self._drex_output_parameters.surprisal):
            value = row[0] # assumption: length == 1, i.e. only one feature
            data.at[idx, "drex_surprisal"] = value

        for idx, row in enumerate(self._drex_output_parameters.context_beliefs):
            data.at[idx, "drex_context_beliefs"] = row
            
        for idx, row in enumerate(self._drex_output_parameters.psi):
            data.at[idx, "drex_predictions"] = row
            entropy = self._drex_output_parameters.entropy_of(row)
            data.at[idx, "drex_entropy"] = entropy

        for idx, row in enumerate(self._drex_output_parameters.change_decision_probability):
            data.at[idx, "drex_cd_probability"] = row
        
        for idx, row in enumerate(self._drex_output_parameters.belief_dynamics):
            data.at[idx, "drex_bd"] = row

        return data

    def write_mat(self, filename):
        df = self.df.drop(self.df.tail(1).index)
        mat_data = {
            "ppm_output": str(self._ppm_output_parameters.source_file_path),
            "drex_output": str(self._drex_output_parameters.source_file_path),
            "df": {name: col.values for name, col in df.items()}
        }
        sio.savemat(filename,  mat_data)
        return filename