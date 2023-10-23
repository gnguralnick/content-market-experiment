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

import scipy.optimize as opt

def minimize_with_retry(fun, x0, args, constraints=None, bounds=None, tol=None, num_retry=1):
    result = opt.minimize(
        fun=fun,
        x0=x0,
        args=args,
        constraints=constraints,
        bounds=bounds,
        options={'maxiter': 1000},
        tol=tol
    )

    retries = 0
    while not result.success and retries < num_retry:
        result = opt.minimize(
            fun=fun,
            x0=result.x,
            args=args,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000,'maxls': 1000, 'ftol': 2e-9, 'gtol': 1e-8},
            tol=1e-9
        )

        retries += 1

    if not result.success:
        raise RuntimeError('Optimization failed', result.message)
        
    return result