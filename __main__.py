import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from supervised.linear_regression import (
    LearningProcedure,
    LearningProcedureFactory,
    LinearRegression,
)


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
