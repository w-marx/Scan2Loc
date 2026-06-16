import numpy as np
from numbers import Real

class Scheduler:
    """
    Shedules which one of n options to choose.
    sucess will make it more likely that get_best will return it again.
    failure will do the opposite
    """
    def __init__(self, n_options:int):
        assert n_options > 0
        self.n_options = n_options

        self.num_successes = np.zeros(self.n_options, np.uint64)
        self.num_failures = np.zeros(self.n_options, np.uint64)

    def __str__(self):
        return "Always first image Scheduler"

    def get_best(self)->int:
        return 0
    
    def adjust(self, i:int, success:bool):
        assert i < self.n_options
    
        if success:
            self.num_successes[i] += 1
        else:
            self.num_failures[i] += 1

    def get_index_success_ratios(self)->list[tuple[int, Real]]:
        N = self.num_successes[:] + self.num_failures[:]
        ratios = self.num_failures[:]/N[:]

        return [(i, ratio) for i, ratio in enumerate(ratios)]


    
class EMAScheduler(Scheduler):
    """
    Shedules which one of n options to choose.
    New score is computed by: (1-alpha) * p[i] + alpha * [sucess | failure]
    """
    def __init__(self, n_options:int, alpha:float = 0.05):
        super().__init__(n_options = n_options)
        self.alpha = alpha
        self.probabilities = np.ones(n_options, dtype = np.float32)

    def __str__(self):
        return "EMA Scheduler"

    def adjust(self, i:int, success:bool):
        target = 1.0 if success else 0.0

        self.probabilities[i] = (
            (1 - self.alpha) * self.probabilities[i]
            + self.alpha * target
        )

        super().adjust(i, success)
    
    def get_best(self)->int:
        return int(np.argmax(self.probabilities))
    


class BlockingEMAScheduler(Scheduler):
    """
    Scheduler for SLAM reference image selection.
    - Success increases probability for repeated selection (momentum)
    - Failed indices are blocked until another index succeeds
    """
    def __init__(self, n_options: int, alpha: float = 0.3, momentum: float = 0.5):
        super().__init__(n_options=n_options)
        self.alpha = alpha
        self.momentum = momentum

        self.probabilities = np.ones(n_options, dtype=np.float32) / n_options
        self.blocked = np.zeros(n_options, dtype=bool)
        self.last_success_idx = None
    
    def __str__(self):
        return "Blocking EMA Scheduler"
        
    def adjust(self, i: int, success: bool):
        if success:
            self.probabilities[i] = ((1 - self.alpha) * self.probabilities[i]+ self.alpha )
            self.blocked[:] = False
        else:
            self.blocked[i] = True
            self.probabilities[i] = 0.0
        
        self.probabilities /= np.sum(self.probabilities)        
        super().adjust(i, success)
    
    def get_best(self) -> int:
        if np.all(self.blocked):
            self.blocked[:] = False
            self.probabilities = np.ones(len(self.probabilities), dtype=np.float32) / len(self.probabilities)
        
        masked_probs = np.where(self.blocked, -np.inf, self.probabilities)
        return int(np.argmax(masked_probs))