from dataclasses import dataclass
from typing import Callable, List

from utils.vectors import multiply_by_number, subtract, vsum


@dataclass
class GradientDescent:
    h: Callable[[List[float]], float]
    alpha: float
    iterations: int

    def __call__(self, X, Y):
        m = len(X)
        T = [0] * len(X[0])

        for _ in range(self.iterations):
            D = multiply_by_number(
                vsum(
                    [multiply_by_number(x, self.h(T)(x) - y) for x, y in zip(X, Y)]
                ),
                1 / m,
            )
            T = subtract(T, multiply_by_number(D, self.alpha))

        return T
