from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matrices import mul, transpose
from vectors import dot, multiply_by_number, subtract, vsum


class NormalEquation:
    def __call__(self, X, Y):
        Y = [[y] for y in Y]
        return [
            l[0]
            for l in mul(
                mul(np.linalg.inv(mul(transpose(X), X)), transpose(X)), Y
            )
        ]


@dataclass
class GradientDescent:
    alpha: float
    iterations: int

    @staticmethod
    def h(T):
        """Hypothesis"""
        return lambda X: dot(T, X)

    def __call__(self, X, Y):
        m = len(X)
        T = [0] * len(X[0])

        for i in range(self.iterations):
            D = multiply_by_number(
                vsum(
                    [
                        multiply_by_number(x, self.h(T)(x) - y)
                        for x, y in zip(X, Y)
                    ]
                ),
                1 / m,
            )
            T = subtract(T, multiply_by_number(D, self.alpha))

        return T


class LearningProcedure(Enum):
    GRADIENT_DESCENT = 1
    NORMAL_EQUATION = 2


class LearningProcedureFactory:
    @classmethod
    def create_learning_procedure(
        cls, learning_procedure, alpha=0.0001, iterations=1000
    ):
        if learning_procedure == LearningProcedure.GRADIENT_DESCENT:
            return GradientDescent(alpha, iterations)
        elif learning_procedure == LearningProcedure.NORMAL_EQUATION:
            return NormalEquation()


class LinearRegression:
    """Linear regression with multiple features"""

    def __init__(self, X, Y, learning_procedure):
        self._X = [(1, *x) for x in X]
        self._Y = Y
        self._m = len(X)
        self.predict = None
        self._learning_procedure = learning_procedure
        self._T = [0] * len(X[0])

    @staticmethod
    def h(T):
        """Hypothesis"""
        return lambda X: dot(T, X)

    @property
    def _c(self):
        """Cost function"""
        return sum(
            (self.h(self._T)(x) - y) ** 2 for x, y in zip(self._X, self._Y)
        ) / (2 * self._m)

    def train(self):
        """Train a model"""
        self._T = self._learning_procedure(self._X, self._Y)
        self.predict = self.h(self._T)


if __name__ == "__main__":
    data = pd.read_csv("student.csv")

    X1, X2 = data["Math"].values, data["Reading"].values
    Y = data["Writing"].values

    lp = LearningProcedureFactory.create_learning_procedure(
        LearningProcedure.NORMAL_EQUATION
    )
    lr = LinearRegression(list(zip(X1, X2)), Y, lp)

    lr.train()

    print(f"{lr._T=} {lr._c=}")

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X1, X2, Y, color="#ef1234")
    plt.show()

    print(lr.predict((1, 60, 70)))
