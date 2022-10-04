import numpy as np
import pandas as pd
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, n):
    it = []
    coste = []
    m = len(X)

    for i in range(0, n):
        h = np.dot(X, theta)
        theta = theta - np.dot(X.T, h - y) * (1 / m) * alpha
        it.append(i)
        coste.append(computeCost(X, y, theta))

    J_history = pd.DataFrame({'iteracion': it, 'coste': coste})
    return J_history, theta
