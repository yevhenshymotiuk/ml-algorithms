import math

from enum import Enum


class LearningProcedure(Enum):
    ORDINARY_LEAST_SQUARE = 1
    GRADIENT_DESCENT = 2


class SimpleLinearRegression:
    """Simple linear regression implementation"""

    def __init__(self, training_set, learning_procedure):
        self._training_set = training_set
        self._m = len(training_set)
        self.predict = None
        self.learning_procedure = learning_procedure

        if learning_procedure == LearningProcedure.GRADIENT_DESCENT:
            self._alpha = 0.01
            self._iterations = 1000

    @staticmethod
    def _h(t0, t1):
        """Hypothesis"""
        return lambda x: t0 + t1 * x

    def _c(self, t0, t1):
        """Cost function"""
        return sum(
            (self._h(t0, t1)(x) - y) ** 2
            for x, y in self._training_set.items()
        ) / (2 * self._m)

    def _gd(self):
        """Gradient descent"""
        t0, t1 = 0, 0

        for _ in range(self._iterations):
            tmp0 = (
                t0
                - self._alpha
                * sum(
                    (self._h(t0, t1)(x) - y)
                    for x, y in self._training_set.items()
                )
                / self._m
            )
            tmp1 = (
                t1
                - self._alpha
                * sum(
                    (self._h(t0, t1)(x) - y) * x
                    for x, y in self._training_set.items()
                )
                / self._m
            )
            t0 = tmp0
            t1 = tmp1

        return self._h(t0, t1)

    def _rmse(self, t0, t1):
        """Root mean squared error"""
        return math.sqrt(
            sum(
                (self._h(t0, t1)(x) - y) ** 2
                for x, y in self._training_set.items()
            )
            / self._m
        )

    def _ols(self):
        "Ordinary least square"
        x_mean = sum(self._training_set.keys()) / self._m
        y_mean = sum(self._training_set.values()) / self._m

        t1 = sum(
            (x - x_mean) * (y - y_mean) for x, y in self._training_set.items()
        ) / sum((x - x_mean) ** 2 for x in self._training_set)
        t0 = y_mean - t1 * x_mean

        return self._h(t0, t1)

    def train(self):
        """Train a model"""
        if self.learning_procedure == LearningProcedure.ORDINARY_LEAST_SQUARE:
            self.predict = self._ols()
        elif self.learning_procedure == LearningProcedure.GRADIENT_DESCENT:
            self.predict = self._gd()


if __name__ == "__main__":
    ts = {
        1: 1,
        2: 4,
        3: 9,
        4: 16,
        5: 25,
        7: 49,
        8: 64,
        9: 81,
        10: 100,
        11: 121,
        12: 144,
        13: 149,
        14: 196,
    }

    lr = SimpleLinearRegression(ts, LearningProcedure.ORDINARY_LEAST_SQUARE)

    lr.train()

    print(lr.predict(6))
