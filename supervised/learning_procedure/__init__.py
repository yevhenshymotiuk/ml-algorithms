from enum import Enum

from .gradient_descent import GradientDescent
from .normal_equation import NormalEquation


class LearningProcedure(Enum):
    GRADIENT_DESCENT = 1
    NORMAL_EQUATION = 2


class LearningProcedureFactory:
    @classmethod
    def create_learning_procedure(
        cls, learning_procedure, h=None, alpha=0.0001, iterations=1000
    ):
        if learning_procedure == LearningProcedure.GRADIENT_DESCENT:
            if not h:
                raise TypeError("h parameter is required")

            return GradientDescent(h, alpha, iterations)

        return NormalEquation()
