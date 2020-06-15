# pylint: disable=logging-fstring-interpolation, broad-except
"""common"""
#import signal
import math
import time
from contextlib import contextmanager
import numpy as np
import yaml
from common import get_logger

VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)

CUM = 0
RESET = 1
MODES = set([CUM, RESET])


OP_MAP = {
    'mean': np.mean,
    'max': np.max,
    'std': np.std,
    'sum': sum,
}


class TimeoutException(Exception):
    """timeoutexception"""


class Timer:
    """timer"""
    def __init__(self):
        self.total = {}
        self.history = {}
        self.modes = {}

    @classmethod
    def from_file(cls, save_file):
        """contruct timer from a save file"""
        timer = Timer()
        timer.load(save_file)
        return timer

    def add_process(self, pname, time_budget, mode=RESET):
        """set time_budget
        mode: CUM/RESET
        """
        if pname in self.total:
            raise ValueError(f"Existing process of timer: {pname}")
        if mode not in MODES:
            raise ValueError(f"wrong process mode: {mode}")

        self.total[pname] = time_budget
        self.history[pname] = []
        self.modes[pname] = mode

    @contextmanager
    def time_limit(self, pname, verbose=True):
        """limit time"""
        def signal_handler(signum, frame):
            raise TimeoutException(f"{pname}: Timed out!")
#        signal.signal(signal.SIGALRM, signal_handler)
        time_budget = int(math.ceil(self.get_remain(pname)))
#        signal.alarm(time_budget)
        start_time = time.time()

        try:

            if verbose:
                LOGGER.info(f'start {pname} with time budget {time_budget}')
            yield
        finally:
            exec_time = time.time() - start_time
#            signal.alarm(0)
            self.history[pname].append(exec_time)

        if verbose:
            LOGGER.info(f'{pname} success, time spent {exec_time} sec')

        if self.get_remain(pname) <= 0:
            raise TimeoutException(f"{pname}: Timed out!")

    def get_remain(self, pname):
        """get remaining time of process"""
        if self.modes[pname] == CUM:
            remain = self.total[pname] - sum(self.history[pname])
        else:
            remain = self.total[pname]

        return remain

    def get_all_remain(self):
        """get remaining time of process"""
        return {key: self.get_remain(key) for key in self.total.keys()}

    def get_stats(self, pname):
        """get stats of timing history"""
        result = {}
        for stat in ['sum', 'mean', 'max', 'std']:
            history = self.history[pname]
            if history:
                result[stat] = float(OP_MAP[stat](self.history[pname]))
            else:
                result[stat] = 0
        return result

    def get_overall_duration(self):
        """get overall duration"""
        duration = 0
        for _, value in self.history.items():
            duration += sum(value)
        return duration

    def get_all_stats(self):
        """get all stats of timing history"""
        stats = {pname: self.get_stats(pname) for pname in self.total.keys()}
        return stats

    def save(self, save_file):
        """save timer"""
        save_content = {
            'total': self.total,
            'history': self.history,
            'modes': self.modes
        }
        with open(save_file, 'w') as ftmp:
            yaml.dump(save_content, ftmp)

    def load(self, save_file):
        """load timer"""
        with open(save_file, 'r') as ftmp:
            save_content = yaml.safe_load(ftmp)
        self.total = save_content['total']
        self.history = save_content['history']
        self.modes = save_content['modes']
