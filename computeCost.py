import numpy as np


def computeCost(X, y, theta):
    m = len(X)

    h = np.dot(X, theta)

    cuadrado = np.power(h - y, 2)  # ** para elevar

    coste = (np.sum(cuadrado) / (2 * m))  # - == np.substract(,)

    return coste
