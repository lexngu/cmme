import pymatbridge as pymat
from datetime import datetime
import time
import threading
from pathlib import Path
from .config import Config


class MatlabWorker:
    _matlab_instance = None
    _matlab_instance_running = False
    _matlab_last_action = None
    _matlab_work_in_progress = 0

    AUTOSTOP_WAIT_TIME = 10  # seconds
    DREX_ESTIMATE_PRIOR_SCRIPT_PATH = (Path( __file__ ).parent.absolute() / "./res/matlab/drex_estimate.m").resolve()
    DREX_RUN_SCRIPT_PATH = (Path( __file__ ).parent.absolute() / "./res/matlab/drex_run.m").resolve()
    SUMMARY_PLOT_SCRIPT_PATH = (Path( __file__ ).parent.absolute() / "./res/matlab/summary_plot.m").resolve()

    def estimate_prior(self, input_file_path):
        """Triggers the execution of the wrapper script, running D-REX's estimate_suffstat.m function."""
        return MatlabWorker._run_func(MatlabWorker.DREX_ESTIMATE_PRIOR_SCRIPT_PATH, input_file_path)

    def run_model(self, input_file_path):
        """Triggers the execution of the wrapper script, running D-REX's run_DREX_model.m function."""
        return MatlabWorker._run_func(MatlabWorker.DREX_RUN_SCRIPT_PATH, input_file_path)

    def plot(self, input_file_path):
        """Triggers the execution of the script generating the comparison plot."""
        return MatlabWorker._run_func(MatlabWorker.SUMMARY_PLOT_SCRIPT_PATH, input_file_path)

    def _run_func(func_path, *func_args, **kwargs):
        MatlabWorker._autostart_matlab()
        MatlabWorker._matlab_work_in_progress += 1
        result = MatlabWorker._matlab_instance.run_func(func_path, *func_args, **kwargs)
        MatlabWorker._matlab_work_in_progress -= 1
        return result

    def _run_code(code):
        MatlabWorker._autostart_matlab()
        MatlabWorker._matlab_work_in_progress += 1
        result = MatlabWorker._matlab_instance.run_code(code)
        MatlabWorker._matlab_work_in_progress -= 1
        return result

    def _start_matlab():
        if MatlabWorker._matlab_instance == None:
            MatlabWorker._matlab_instance = pymat.Matlab(executable=Config().matlab_path(),
                                                         startup_options="-nodisplay -nodesktop -nosplash")
            MatlabWorker._matlab_instance.start()
            MatlabWorker._matlab_instance_running = True
            MatlabWorker._autostop_thread = threading.Thread(target=MatlabWorker._autostop_matlab_thread_func)
            MatlabWorker._autostop_thread.start()

    def _stop_matlab():
        if MatlabWorker._matlab_instance != None:
            MatlabWorker._matlab_instance.stop()
            MatlabWorker._matlab_instance_running = False
            MatlabWorker._matlab_instance = None

    def restart_matlab():
        MatlabWorker._stop_matlab()
        MatlabWorker._start_matlab()

    def _autostart_matlab():
        MatlabWorker._matlab_last_action = datetime.now()
        if MatlabWorker._matlab_instance_running != True:
            MatlabWorker._start_matlab()

    def _autostop_matlab_thread_func():
        sleep_time = max(MatlabWorker.AUTOSTOP_WAIT_TIME / 5.0, 1)
        while (MatlabWorker._matlab_instance != None):
            now = datetime.now()
            if (
                    now - MatlabWorker._matlab_last_action).total_seconds() >= MatlabWorker.AUTOSTOP_WAIT_TIME and MatlabWorker._matlab_work_in_progress <= 0:
                MatlabWorker._stop_matlab()
            time.sleep(sleep_time)
