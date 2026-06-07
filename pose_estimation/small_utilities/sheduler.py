import numpy as np

class Sheduler:
    """
    Shedules which one of n options to choose.
    sucess will make it more likely that get_best will return it again.
    failure will do the opposite
    """
    def __init__(self, n_options:int):
        assert n_options > 0
        self.n_options = n_options

    def get_best(self)->int:
        return 0
    
    def adjust(self, i, sucess:bool):
        assert i < self.n_options
    
        if sucess:
            self.sucess(i)
        else:
            self.failure(i)
    
class EMASheduler(Sheduler):
    """
    Shedules which one of n options to choose.
    New score is computed by: (1-alpha) * p[i] + alpha * [sucess | failure]
    """
    def __init__(self, n_options:int, alpha:float = 0.05):
        assert n_options > 0
        self.n_options = n_options
        self.alpha = alpha
        self.probabilities = np.ones(n_options, dtype = np.float32)

    def adjust(self, i, success:bool):
        target = 1.0 if success else 0.0

        self.probabilities[i] = (
            (1 - self.alpha) * self.probabilities[i]
            + self.alpha * target
        )
    
    def get_best(self)->int:
        return np.argmax(self.probabilities)