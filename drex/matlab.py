import threading
from datetime import datetime
import time
from pathlib import Path
import numpy as np

import matlab.engine

from .distribution import Distribution
from .instructions_file import InstructionsFile
from .model import DREXInstance
from .prior import UnprocessedPrior
from .results_file import ResultsFile, parse_post_DREX_prediction_results


class MatlabWorker:
    _matlab_engine = None
    _matlab_engine_running = False
    _matlab_last_action = None
    _matlab_work_in_progress = 0

    AUTOSTOP_WAIT_TIME = 10  # seconds

    DREX_INTERMEDIATE_SCRIPT_PATH = (Path(__file__).parent.absolute() / "../res/wrappers/d-rex/drex_intermediate_script.m").resolve()
    SUMMARY_PLOT_SCRIPT_PATH = (Path( __file__ ).parent.absolute() / "../res/wrappers/d-rex/summary_plot.m").resolve()

    def run_model(self, input_file_path: Path):
        """Triggers the execution of the wrapper script, running D-REX's run_DREX_model.m function."""
        MatlabWorker._matlab_last_action = datetime.now()
        MatlabWorker._autostart_matlab()
        MatlabWorker._matlab_work_in_progress += 1
        MatlabWorker._matlab_engine.addpath(str(MatlabWorker.DREX_INTERMEDIATE_SCRIPT_PATH.parent))
        result = MatlabWorker._matlab_engine.drex_intermediate_script(input_file_path)
        MatlabWorker._matlab_work_in_progress -= 1
        return result

    def plot(self, input_file_path: Path):
        """Triggers the execution of the script generating the comparison plot."""
        MatlabWorker._matlab_last_action = datetime.now()
        MatlabWorker._autostart_matlab()
        MatlabWorker._matlab_work_in_progress += 1
        MatlabWorker._matlab_engine.addpath(str(MatlabWorker.SUMMARY_PLOT_SCRIPT_PATH.parent))
        result = MatlabWorker._matlab_engine.summary_plot(input_file_path)
        MatlabWorker._matlab_work_in_progress -= 1
        return result

    def to_mat(self, data, path):
        """

        :param data: dict containing matlab-compatible data
        :param path:
        :return:
        """
        MatlabWorker._autostart_matlab()
        MatlabWorker._matlab_work_in_progress += 1
        MatlabWorker._matlab_last_action = datetime.now()
        for k, v in data.items():
            MatlabWorker._matlab_engine.workspace[k] = v
        keys = data.keys()
        MatlabWorker._matlab_engine.save(str(path), *keys, nargout=0)
        MatlabWorker._matlab_work_in_progress -= 1
        return path

    def from_mat(self, path):
        MatlabWorker._autostart_matlab()
        MatlabWorker._matlab_work_in_progress += 1
        MatlabWorker._matlab_last_action = datetime.now()

        data = {}
        MatlabWorker._matlab_engine.load(path, nargout=0)
        varnames = MatlabWorker._matlab_engine.who()
        for v in varnames:
            data[v] = MatlabWorker._matlab_engine.workspace[v]

        MatlabWorker._matlab_work_in_progress -= 1
        return data

    def _start_matlab():
        if MatlabWorker._matlab_engine == None:
            MatlabWorker._matlab_engine = matlab.engine.start_matlab()
            MatlabWorker._matlab_engine_running = True
            MatlabWorker._autostop_thread = threading.Thread(target=MatlabWorker._autostop_matlab_thread_func)
            MatlabWorker._autostop_thread.start()

    def _stop_matlab():
        if MatlabWorker._matlab_engine != None:
            MatlabWorker._matlab_engine.exit()
            MatlabWorker._matlab_engine_running = False
            MatlabWorker._matlab_engine = None

    def restart_matlab():
        MatlabWorker._stop_matlab()
        MatlabWorker._start_matlab()

    def _autostart_matlab():
        MatlabWorker._matlab_last_action = datetime.now()
        if MatlabWorker._matlab_engine_running != True:
            MatlabWorker._start_matlab()

    def _autostop_matlab_thread_func():
        sleep_time = max(MatlabWorker.AUTOSTOP_WAIT_TIME / 5.0, 1)
        while (MatlabWorker._matlab_engine != None):
            now = datetime.now()
            if (
                    now - MatlabWorker._matlab_last_action).total_seconds() >= MatlabWorker.AUTOSTOP_WAIT_TIME and MatlabWorker._matlab_work_in_progress <= 0:
                MatlabWorker._stop_matlab()
            time.sleep(sleep_time)


def write_instructions_file(instructions_file: InstructionsFile, instructions_file_path):
    data = dict()
    p = instructions_file.prior

    # Add instructions for procesing UnprocessedPrior by using estimate_suffstat.m
    if isinstance(p, UnprocessedPrior):
        data["estimate_suffstat"] = {
            "xs": matlab.double(p._prior_input_sequence),
            "params": {
                "distribution": p.distribution().value,
                "D": p.D()
            }
        }
        if p.distribution() == Distribution.GMM:
            data["estimate_suffstat"]["params"]["max_ncomp"] = p._max_n_comp

    # Add instructions for invoking D-REX (run_DREX_model.m)
    data["run_DREX_model"] = {
        "x": matlab.double(instructions_file.input_sequence),
        "params": {
            "distribution": instructions_file.distribution.value,
            "D": instructions_file.D,
            "hazard": matlab.double(instructions_file.hazard),
            "obsnz": matlab.double(instructions_file.obsnz),
            "memory": instructions_file.memory,
            "maxhyp": instructions_file.maxhyp
        },
    }

    # Add instructions for post_DREX_changedecision.m
    data["post_DREX_changedecision"] = {
        "threshold": instructions_file.change_decision_threshold
    }

    # Add results_file_path
    data["results_file_path"] = instructions_file.results_file_path

    # Write and return
    return MatlabWorker().to_mat(data, instructions_file_path)


def write_results_file(results_file):
    pass


def invoke_model(drex_instance: DREXInstance):
    """
    Writes the instructions file, invokes the model, and returns the results file.
    :return: ResultsFile
    """
    instructions_file_path = "instru.mat"
    results_file_path = "resu.mat"

    instructions_file = InstructionsFile(drex_instance._distribution, drex_instance._D, drex_instance._prior, drex_instance._hazard,
                                         drex_instance._observation_noise, drex_instance._memory, drex_instance._maxhyp,
                                         drex_instance._change_decision_threshold, drex_instance._input_sequence, results_file_path)
    instructions_file_path = write_instructions_file(instructions_file, instructions_file_path)

    mw = MatlabWorker()
    mw_results = mw.run_model(instructions_file_path)

    results_file = parse_results_file(results_file_path)

    return results_file


def parse_results_file(file_path):
    mw = MatlabWorker()
    data = mw.from_mat(file_path)

    instructions_file_path = data["instructions_file_path"]
    input_sequence = data["input_sequence"]
    run_results = data["run_DREX_model_results"]
    bd_results = data["post_DREX_beliefdynamics_results"]
    cd_results = data["post_DREX_changedecision_results"]
    pred_results = data["post_DREX_prediction_results"]

    input_sequence = np.array(input_sequence)
    surprisal = np.array(run_results["surprisal"])
    joint_surprisal = np.array(run_results["joint_surprisal"])
    context_beliefs = np.array(run_results["context_beliefs"])
    belief_dynamics = np.array(bd_results)
    change_decision_changepoint = cd_results["changepoint"]
    change_decision_probability = np.array(cd_results["changeprobability"])
    psi = parse_post_DREX_prediction_results(pred_results)

    return ResultsFile(instructions_file_path, input_sequence, surprisal, joint_surprisal, context_beliefs, belief_dynamics, change_decision_changepoint, change_decision_probability, psi)


def parse_instructions_file(file_path):
    pass

