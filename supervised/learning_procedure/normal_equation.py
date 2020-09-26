from utils.matrices import mul, transpose

import numpy as np


class NormalEquation:
    def __call__(self, X, Y):
        Y = [[y] for y in Y]
        return [
            l[0]
            for l in mul(
                mul(np.linalg.inv(mul(transpose(X), X)), transpose(X)), Y
            )
        ]
