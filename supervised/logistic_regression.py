import math
from decimal import Decimal

from utils.vectors import dot


class LogisticRegression:
    """Logistic regression with two classes"""

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
        g = lambda z: 1 / 1 - math.e ** -z

        return lambda X: g(dot(T, X))

    @property
    def _c(self):
        """Cost function"""
        return (
            sum(
                y * math.log2(self.h(self._T)(x))
                + (1 - y) * math.log2(1 - self.h(self._T)(x))
                for x, y in zip(self._X, self._Y)
            )
            / -self._m
        )

    def train(self):
        """Train a model"""
        self._T = self._learning_procedure(self._X, self._Y)
        self.predict = self.h(self._T)
