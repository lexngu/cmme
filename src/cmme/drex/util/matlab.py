import threading
from datetime import datetime
import time
from pathlib import Path
# Use scipy, since matlabengine does not support "struct"s, which are originally created by D-REX's functions.
import scipy.io as sio

import matlab.engine
from pymatbridge import pymatbridge

from cmme.config import Config

def to_mat(data, file_path):
    sio.savemat(str(file_path), data)
    return file_path

def from_mat(file_path):
    mat_data = sio.loadmat(file_path)
    return mat_data

class MatlabWorker:
    DREX_INTERMEDIATE_SCRIPT_PATH = (Path(
        __file__).parent.parent.parent.parent.parent.absolute() / "./res/wrappers/d-rex/drex_intermediate_script.m").resolve()
    SUMMARY_PLOT_SCRIPT_PATH = (Path(
        __file__).parent.parent.parent.parent.parent.absolute() / "./res/wrappers/d-rex/summary_plot.m").resolve()

    AUTOSTOP_WAIT_TIME = 10 # seconds

    def run_model(instructions_file_path: Path):
        """
        Triggers the execution of the wrapper script, running D-REX's run_DREX_model.m function.
        :return: dictionary with MATLAB output
        """
        MatlabEngineWorker._autostart_matlab()
        MatlabEngineWorker._matlab_work_in_progress += 1

        MatlabEngineWorker._matlab_engine.addpath(str(MatlabWorker.DREX_INTERMEDIATE_SCRIPT_PATH.parent))  # load script
        result = MatlabEngineWorker._matlab_engine.drex_intermediate_script(str(instructions_file_path))  # execute script

        MatlabEngineWorker._matlab_work_in_progress -= 1
        return result

    def plot(input_file_path: Path):
        """Triggers the execution of the script generating the comparison plot."""
        PymatbridgeMatlabWorker._autostart_matlab()
        PymatbridgeMatlabWorker._matlab_work_in_progress += 1

        PymatbridgeMatlabWorker._matlab_instance.addpath(str(MatlabWorker.SUMMARY_PLOT_SCRIPT_PATH.parent)) # load script
        result = PymatbridgeMatlabWorker._matlab_instance.summary_plot(str(input_file_path)) # execute script

        PymatbridgeMatlabWorker._matlab_work_in_progress -= 1
        return result

class MatlabEngineWorker:
    _matlab_engine = None
    _matlab_engine_running = False
    _matlab_last_action = None
    _matlab_work_in_progress = 0

    def _start_matlab():
        if MatlabEngineWorker._matlab_engine == None:
            MatlabEngineWorker._matlab_engine = matlab.engine.start_matlab()
            MatlabEngineWorker._matlab_engine_running = True
            MatlabEngineWorker._autostop_thread = threading.Thread(target=MatlabEngineWorker._autostop_matlab_thread_func)
            MatlabEngineWorker._autostop_thread.start()

    def _stop_matlab():
        if MatlabEngineWorker._matlab_engine != None:
            MatlabEngineWorker._matlab_engine.exit()
            MatlabEngineWorker._matlab_engine_running = False
            MatlabEngineWorker._matlab_engine = None

    def restart_matlab():
        MatlabEngineWorker._stop_matlab()
        MatlabEngineWorker._start_matlab()

    def _autostart_matlab():
        MatlabEngineWorker._matlab_last_action = datetime.now()
        if MatlabEngineWorker._matlab_engine_running != True:
            MatlabEngineWorker._start_matlab()

    def _autostop_matlab_thread_func():
        sleep_time = max(MatlabWorker.AUTOSTOP_WAIT_TIME / 5.0, 1) # seconds until next check; once every >=1s
        while (MatlabEngineWorker._matlab_engine != None):
            now = datetime.now()
            if MatlabEngineWorker._matlab_work_in_progress <= 0 and\
                    (now - MatlabEngineWorker._matlab_last_action).total_seconds() >= MatlabWorker.AUTOSTOP_WAIT_TIME:
                MatlabEngineWorker._stop_matlab()
            time.sleep(sleep_time)

class PymatbridgeMatlabWorker:
    _matlab_instance = None
    _matlab_instance_running = False
    _matlab_last_action = None
    _matlab_work_in_progress = 0

    def _start_matlab(matlab_executable_path = str(Config().matlab_path())):
        if PymatbridgeMatlabWorker._matlab_instance == None:
            PymatbridgeMatlabWorker._matlab_instance = pymatbridge.Matlab(executable=matlab_executable_path, startup_options="-nodisplay -nodesktop -nosplash")
            PymatbridgeMatlabWorker._matlab_instance.start()
            PymatbridgeMatlabWorker._matlab_instance_running = True
            PymatbridgeMatlabWorker._autostop_thread = threading.Thread(target=PymatbridgeMatlabWorker._autostop_matlab_thread_func)
            PymatbridgeMatlabWorker._autostop_thread.start()

    def _stop_matlab():
        if PymatbridgeMatlabWorker._matlab_instance != None:
            PymatbridgeMatlabWorker._matlab_instance.exit()
            PymatbridgeMatlabWorker._matlab_instance_running = False
            PymatbridgeMatlabWorker._matlab_instance = None

    def restart_matlab():
        PymatbridgeMatlabWorker._stop_matlab()
        PymatbridgeMatlabWorker._start_matlab()

    def _autostart_matlab():
        PymatbridgeMatlabWorker._matlab_last_action = datetime.now()
        if PymatbridgeMatlabWorker._matlab_instance_running != True:
            PymatbridgeMatlabWorker._start_matlab()

    def _autostop_matlab_thread_func():
        sleep_time = max(MatlabWorker.AUTOSTOP_WAIT_TIME / 5.0, 1) # seconds until next check; once every >=1s
        while (PymatbridgeMatlabWorker._matlab_instance != None):
            now = datetime.now()
            if PymatbridgeMatlabWorker._matlab_work_in_progress <= 0 and\
                    (now - PymatbridgeMatlabWorker._matlab_last_action).total_seconds() >= MatlabWorker.AUTOSTOP_WAIT_TIME:
                PymatbridgeMatlabWorker._stop_matlab()
            time.sleep(sleep_time)