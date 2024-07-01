from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import ruptures as rpt
from ruptures.base import BaseCost

# Parameters for CP detection, empirically determined
JUMP = 1
PENALTY = 5
THRESHOLD = 15

class CostFunction(Enum):
    """
    The available cost functions
    """
    L1 = rpt.costs.CostL1()
    L2 = rpt.costs.CostL2()
    NORMAL = rpt.costs.CostNormal()
    RBF = rpt.costs.CostRbf()
    COS = rpt.costs.CostCosine()
    LIN = rpt.costs.CostLinear()
    CLIN = rpt.costs.CostCLinear()
    RANK = rpt.costs.CostRank()
    MI = rpt.costs.CostMl()
    AR = rpt.costs.CostAR()

class CpDetAlgo(ABC):
    """
    Baseclass for a changepoint detection algorithm.
    """

    def __init__(self, cost: CostFunction):
        self.cost = cost.value

    def find_cp(
        self,
        signal: np.ndarray,
        jump: int = JUMP,
        pen: float = PENALTY,
        threshold: float = THRESHOLD,
    ) -> np.ndarray:
        """
        Run the algorithm on the input signal.
        Baseclass outputs a numpy array of zeros with length n_changepoints.

        :param signal:
        :return: a list with indices of the detected changepoints.
        """
        self.jump = jump
        self.pen = pen
        self.threshold = threshold

        cps = self.algorithm(signal, self.cost)  # perform algorithm
        return cps

    @abstractmethod
    def algorithm(self, signal: np.ndarray, cost: BaseCost) -> list:
        """
        Actual algorithm used
        :param signal: signal to evaluate
        :return: a list with indices of the detected changepoints.
        """
        print("No change-point algorithm defined.")
        pass


class BinarySegmentation(CpDetAlgo):
    """BinSeg algorithm"""

    def algorithm(self, signal: np.ndarray, cost: BaseCost) -> list:
        self.algo = rpt.Binseg(custom_cost=cost, jump=self.jump).fit(signal)
        cps = self.algo.predict(pen=self.pen, epsilon=self.threshold)
        return cps[:-1]

class Pelt(CpDetAlgo):
    """ PELT algorithm """

    def algorithm(self, signal: np.ndarray, cost: BaseCost) -> list:
        self.algo = rpt.Pelt(custom_cost=cost, jump=self.jump).fit(signal)
        cps = self.algo.predict(pen=self.pen)
        return cps[:-1]


def changepoint_detection(
        samples: dict[int, pd.Series],
        method: CpDetAlgo = Pelt(cost=CostFunction.L2)
) -> dict[int, list[int]]:
    res = {}
    for i, sample in samples.items():
        res[i] = method.find_cp(signal=sample.values)
    return res