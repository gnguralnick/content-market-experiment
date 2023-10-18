import numpy as np
from enum import Enum

# strictly decreasing and continuous functions mapping from all positive numbers to [0, 1]

def exponential_decay(x: float, a: float = 1) -> float:
    return np.exp(-a * x)

def inverse_decay(x: float, a: float = 1, pow: int = 1) -> float:
    return a / (a + x ** pow)

def tanh_decay(x: float, a: float = 1) -> float:
    return 1 - np.tanh(a * x)


class OptimizationTargets(Enum):
    CONSUMER = 1
    PRODUCER = 2
    INFLUENCER = 3