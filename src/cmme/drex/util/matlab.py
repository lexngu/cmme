import threading
from datetime import datetime
import time
from pathlib import Path

import matlab.engine

class MatlabWorker:
    _matlab_engine = None
    _matlab_engine_running = False
    _matlab_last_action = None
    _matlab_work_in_progress = 0

    AUTOSTOP_WAIT_TIME = 10 # seconds

    DREX_INTERMEDIATE_SCRIPT_PATH = (Path(__file__).parent.parent.parent.parent.parent.absolute() / "./res/wrappers/d-rex/drex_intermediate_script.m").resolve()
    SUMMARY_PLOT_SCRIPT_PATH = (Path( __file__ ).parent.parent.parent.parent.parent.absolute() / "./res/wrappers/d-rex/summary_plot.m").resolve()

    def run_model(instructions_file_path: Path):
        """
        Triggers the execution of the wrapper script, running D-REX's run_DREX_model.m function.
        :return: dictionary with MATLAB output
        """
        MatlabWorker._autostart_matlab()
        MatlabWorker._matlab_work_in_progress += 1

        MatlabWorker._matlab_engine.addpath(str(MatlabWorker.DREX_INTERMEDIATE_SCRIPT_PATH.parent)) # load script
        result = MatlabWorker._matlab_engine.drex_intermediate_script(str(instructions_file_path)) # execute script

        MatlabWorker._matlab_work_in_progress -= 1
        return result

    def plot(input_file_path: Path):
        """Triggers the execution of the script generating the comparison plot."""
        MatlabWorker._autostart_matlab()
        MatlabWorker._matlab_work_in_progress += 1

        MatlabWorker._matlab_engine.addpath(str(MatlabWorker.SUMMARY_PLOT_SCRIPT_PATH.parent)) # load script
        result = MatlabWorker._matlab_engine.summary_plot(input_file_path) # execute script

        MatlabWorker._matlab_work_in_progress -= 1
        return result

    def to_mat(data, path):
        """
        Converts a data dictionary to a .mat file, using MATLAB.

        :param data: dict containing matlab-compatible data
        :param path:
        :return: path
        """
        MatlabWorker._autostart_matlab()
        MatlabWorker._matlab_work_in_progress += 1

        for k, v in data.items():
            MatlabWorker._matlab_engine.workspace[k] = v
        keys = data.keys()
        MatlabWorker._matlab_engine.save(str(path), *keys, nargout=0)

        MatlabWorker._matlab_work_in_progress -= 1
        return path

    def from_mat(path : Path) -> dict:
        """
        Parses a .mat file and returns its content as dictionary, using MATLAB.

        :param path:
        :return:
        """
        MatlabWorker._autostart_matlab()
        MatlabWorker._matlab_work_in_progress += 1

        data = {}
        MatlabWorker._matlab_engine.load(str(path), nargout=0)
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
        sleep_time = max(MatlabWorker.AUTOSTOP_WAIT_TIME / 5.0, 1) # seconds until next check; once every >=1s
        while (MatlabWorker._matlab_engine != None):
            now = datetime.now()
            if MatlabWorker._matlab_work_in_progress <= 0 and\
                    (now - MatlabWorker._matlab_last_action).total_seconds() >= MatlabWorker.AUTOSTOP_WAIT_TIME:
                MatlabWorker._stop_matlab()
            time.sleep(sleep_time)
