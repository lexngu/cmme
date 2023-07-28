import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import matlab.engine
from pymatbridge import pymatbridge

from .binding import DREXResultsFile, DREXInstructionsFile
from ..config import Config
from ..lib.model import Model


class MatlabWorker:
    DREX_INTERMEDIATE_SCRIPT_PATH = (Path(
        __file__).parent.parent.parent.parent.absolute() / "./res/wrappers/d-rex/drex_intermediate_script.m").resolve()
    SUMMARY_PLOT_SCRIPT_PATH = (Path(
        __file__).parent.parent.parent.parent.absolute() / "./res/wrappers/d-rex/summary_plot.m").resolve()

    AUTOSTOP_WAIT_TIME = 10  # seconds

    @staticmethod
    def run_model(instructions_file_path: Path):
        """
        Triggers the execution of the wrapper script, running D-REX's run_DREX_model.m function.
        :return: dictionary with MATLAB output
        """
        MatlabEngineWorker.autostart_matlab()
        MatlabEngineWorker.matlab_work_in_progress += 1

        MatlabEngineWorker.matlab_engine\
            .addpath(str(MatlabWorker.DREX_INTERMEDIATE_SCRIPT_PATH.parent))  # load script
        result = MatlabEngineWorker.matlab_engine\
            .drex_intermediate_script(str(instructions_file_path))  # execute script

        MatlabEngineWorker.matlab_work_in_progress -= 1
        return result

    @staticmethod
    def plot(input_file_path: Path):
        """Triggers the execution of the script generating the comparison plot."""
        PymatbridgeMatlabWorker.autostart_matlab()
        PymatbridgeMatlabWorker.matlab_work_in_progress += 1

        PymatbridgeMatlabWorker.matlab_instance\
            .addpath(str(MatlabWorker.SUMMARY_PLOT_SCRIPT_PATH.parent))  # load script
        result = PymatbridgeMatlabWorker.matlab_instance.summary_plot(str(input_file_path))  # execute script

        PymatbridgeMatlabWorker.matlab_work_in_progress -= 1
        return result


class MatlabEngineWorker:
    matlab_engine = None
    _matlab_engine_running = False
    _matlab_last_action = None
    matlab_work_in_progress = 0
    _autostop_thread = None

    @classmethod
    def _start_matlab(cls):
        if cls.matlab_engine is None:
            cls.matlab_engine = matlab.engine.start_matlab()
            cls._matlab_engine_running = True
            cls._autostop_thread = threading.Thread(target=cls._autostop_matlab_thread_func)
            cls._autostop_thread.start()

    @classmethod
    def _stop_matlab(cls):
        if cls.matlab_engine is not None:
            cls.matlab_engine.exit()
            cls._matlab_engine_running = False
            cls.matlab_engine = None

    @classmethod
    def restart_matlab(cls):
        cls._stop_matlab()
        cls._start_matlab()

    @classmethod
    def autostart_matlab(cls):
        cls._matlab_last_action = datetime.now()
        if cls._matlab_engine_running is not True:
            cls._start_matlab()

    @classmethod
    def _autostop_matlab_thread_func(cls):
        sleep_time = max(MatlabWorker.AUTOSTOP_WAIT_TIME / 5.0, 1)  # seconds until next check; once every >=1s
        while cls.matlab_engine is not None:
            now = datetime.now()
            if cls.matlab_work_in_progress <= 0 and\
                    (now - cls._matlab_last_action).total_seconds() >= MatlabWorker.AUTOSTOP_WAIT_TIME:
                cls._stop_matlab()
            time.sleep(sleep_time)


class PymatbridgeMatlabWorker:
    matlab_instance = None
    _matlab_instance_running = False
    _matlab_last_action = None
    matlab_work_in_progress = 0
    _autostop_thread = None

    @classmethod
    def _start_matlab(cls, matlab_executable_path=str(Config().matlab_path())):
        if cls.matlab_instance is None:
            cls.matlab_instance = pymatbridge.Matlab(executable=matlab_executable_path,
                                                     startup_options="-nodisplay -nodesktop -nosplash")
            cls.matlab_instance.start()
            cls._matlab_instance_running = True
            cls._autostop_thread = threading.Thread(target=cls._autostop_matlab_thread_func)
            cls._autostop_thread.start()

    @classmethod
    def _stop_matlab(cls):
        if cls.matlab_instance is not None:
            cls.matlab_instance.exit()
            cls._matlab_instance_running = False
            cls.matlab_instance = None

    @classmethod
    def restart_matlab(cls):
        cls._stop_matlab()
        cls._start_matlab()

    @classmethod
    def autostart_matlab(cls):
        cls._matlab_last_action = datetime.now()
        if cls._matlab_instance_running is not True:
            cls._start_matlab()

    @classmethod
    def _autostop_matlab_thread_func(cls):
        sleep_time = max(MatlabWorker.AUTOSTOP_WAIT_TIME / 5.0, 1)  # seconds until next check; once every >=1s
        while cls.matlab_instance is not None:
            now = datetime.now()
            if cls.matlab_work_in_progress <= 0 and\
                    (now - cls._matlab_last_action).total_seconds() >= MatlabWorker.AUTOSTOP_WAIT_TIME:
                cls._stop_matlab()
            time.sleep(sleep_time)


class DREXModel(Model):
    """
    High-level interface for using D-REX.
    Using +instance+, one can hyper-parameterize D-REX.
    """

    def __init__(self):
        super().__init__()

    def run(self, instructions_file_path) -> DREXResultsFile:
        results = MatlabWorker.run_model(instructions_file_path)
        results_file = DREXResultsFile.load(results['results_file_path'])
        return results_file

    @staticmethod
    def run_instructions_file_at_path(file_path: str) -> DREXResultsFile:
        return DREXModel().run(file_path)
