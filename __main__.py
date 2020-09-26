import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from supervised.learning_procedure import LearningProcedure, LearningProcedureFactory
from supervised.linear_regression import LinearRegression
from supervised.logistic_regression import LogisticRegression


def test_linear_regression():
    data = pd.read_csv("student.csv")

    X1, X2 = data["Math"].values, data["Reading"].values
    Y = data["Writing"].values

    lp = LearningProcedureFactory.create_learning_procedure(
        LearningProcedure.GRADIENT_DESCENT,
        LinearRegression.h,
    )
    lr = LinearRegression(list(zip(X1, X2)), Y, lp)

    lr.train()

    print(f"{lr._T=} {lr._c=}")

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X1, X2, Y, color="#ef1234")
    plt.show()

    print(lr.predict((1, 60, 70)))


def test_logistic_regression():
    data = pd.read_csv("headbrain.csv")

    X1, X2, X3 = data["Age Range"].values, data["Head Size(cm^3)"].values, data["Brain Weight(grams)"].values
    Y = data["Gender"].values

    lp = LearningProcedureFactory.create_learning_procedure(
        LearningProcedure.GRADIENT_DESCENT,
        LogisticRegression.h,
    )
    lr = LogisticRegression(list(zip(X1, X2, X3)), Y, lp)

    lr.train()

    print(lr.predict((1, 1, 3500, 1300)))


test_logistic_regression()
