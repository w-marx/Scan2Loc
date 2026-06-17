import time
import numpy as np
from enum import Enum
from  typing import Union

class TimeLabels(Enum):
    EXTRACT_AND_MATCH_WRAPPER_INIT = "Point based pred. init"
    EXTRACT_AND_MATCH_WRAPPER_CALL = "point based pred."
    PNP_RANSAC = "PnP-Ransac"

    # Line based:
    LSD_AND_CLEANUP = "Generation and cleanup of 2d lines"

class TimeTracker:
    def __init__(self):
        self.last_time = time.perf_counter()
        self.tracked_times = {}
    
    def _get_elapsed_time(self):
        return time.perf_counter()-self.last_time

    def reset_elapsed_time(self):
        """
        Resets the elapsed time, most commonly used before a task marked with add_time_stamp to get
        the clean tim
        """
        self.last_time = time.perf_counter()

    def add_time_stamp(self, name: Union[str, TimeLabels]):
        """
        Adds a time stamp under that name to the tracked times.
        :param name: The name of the timestamp
        """
        if isinstance(name, TimeLabels):
            name = name.value

        time_elapsed = self._get_elapsed_time()
        if name in self.tracked_times:
            self.tracked_times[name].append(time_elapsed)
        else:
            self.tracked_times[name] = [time_elapsed]
        self.reset_elapsed_time()

    
    def return_averaged_times(self)->list[tuple[str, float]]:
        """
        Turns the times dictionary into a list (sorted by avg. time descending)
        :return: A list of ("timestamp_name", "avg_time") tuples, with avg_time in seconds
        """
        avg_times = []
        for key in self.tracked_times:
            avg_times.append(
                (key, np.mean(self.tracked_times[key]))
            )
        return sorted(avg_times, key=lambda x: x[1], reverse=True)


    def get_timestamp_name_times(self, name:str)->None | list[float]:
        """
        Returns the list of times corresponding to the given timestamp name
        :param name: The name of the timestamp (initialised by `add_time_stamp(name)`)
        :return: a list of the times the timestamp took in seconds
        """
        if name in self.tracked_times:
            return self.tracked_times[name]
        else:
            return None


    def get_timestamp_name_avg_time(self, name:str)->None | float:
        """
        Returns the average time of the given timestamp name
        :param name: The name of the timestamp (initialised by `add_time_stamp(name)`)
        :return: The average time of the given timestamp in seconds
        """
        if name in self.tracked_times:
            return np.mean(self.tracked_times[name])
        else:
            return None
    
    def print_report(self):
        """
        Prints each timestamp and its average time sorted
        """
        for key, avg in self.return_averaged_times():
            print(f"{key:<40} {avg*1000:.3f} ms")