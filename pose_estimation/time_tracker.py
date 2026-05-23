import time
import numpy as np

class TimeTracker:
    def __init__(self):
        self.last_time = time.perf_counter()
        self.tracked_times = {}
    
    def _get_elapsed_time(self):
        return time.perf_counter()-self.last_time

    def reset_elapsed_time(self):
        self.last_time = time.perf_counter()

    def add_time_stamp(self, name:str):
        time_ellapsed = self._get_elapsed_time()
        if name in self.tracked_times:
            self.tracked_times[name].append(time_ellapsed)
        else:
            self.tracked_times[name] = [time_ellapsed]
        self.reset_elapsed_time()
    
    def return_averaged_times(self):
        avg_times = []
        for key in self.tracked_times:
            avg_times.append(
                (key, np.mean(self.tracked_times[key]))
            )
        return sorted(avg_times, key=lambda x: x[1], reverse=True)
    
    def print_report(self):
        for key, avg in self.return_averaged_times():
            print(f"{key:<40} {avg*1000:.3f} ms")