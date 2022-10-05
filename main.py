# EPD2: Machine Learning - Regresión

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.linear_model

import plotData
from plotData import *

from computeCost import computeCost

from gradientDescent import gradientDescent


def read_file(file_name):
    # Reading file with data
    print('Loading Data ...', file_name)
    file = pd.read_csv(file_name, names=["poblacion", "beneficio"])
    X = pd.DataFrame({'poblacion': file['poblacion']})
    y = pd.DataFrame({'beneficio': file['beneficio']})

    # Plot data: Note: You have to complete the code in plotData.py
    plot_data(X, y)
    return X, y


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ## ======================= EJ1. Cargar y visualizar =======================
    X, y = read_file('ex1data1.txt')

    ## ======================= EJ2. Función de coste =======================
    print('\nRunning Cost Function ...')
    # Añada una columna con todos sus elementos a 1 a la matriz X como primera columna,
    # e inicializar los parámetros theta a 0

    m = X.shape[0]  # y.shape[0]  ==  len(X)  ==  len(y)

    ones = np.ones((m, 1))

    X['x0'] = ones  # Creo columna x0 y la igualo a ones  # == X['x0'] = 1

    X = X[['x0', 'poblacion']]  # Reordenar

    thetas = np.zeros((X.shape[1], 1))

    J_base = computeCost(X, y, thetas)

    print("\tResult EJ2: Cost = ", J_base)

    ## ======================= EJ3. Gradiente =======================
    # Run gradient descent
    print('\nRunning Gradient Descent ...')
    # Some gradient descent settings
    alpha = 0.01
    it = 1500

    J_history, theta_opt = gradientDescent(X, y, thetas, alpha, it)
    plotData.plotIteracionesVsCoste(J_history)

    print("Theta optima:", theta_opt)

    # print theta to screen

    ## ======================= EJ4. Visualización =======================

    del (X['x0'])
    # Plot the linear fit
    plt.plot(X, theta_opt[1] * X + theta_opt[0])
    plot_data(X, y)

    # Predict values for population sizes of 35, 000 and 70, 000
    x_p = 3.5
    y_p = x_p * theta_opt[1] + theta_opt[0]
    print('X=35,000 -> Y=', y_p)

    plt.scatter(x_p, y_p, marker='^', c='green')
    plt.plot(X, theta_opt[1] * X + theta_opt[0])
    plot_data(X, y)

    x_p = 7
    y_p = x_p * theta_opt[1] + theta_opt[0]
    print('X=70,000 -> Y=', y_p)

    plt.scatter(x_p, y_p, marker='^', c='green')
    plt.plot(X, theta_opt[1] * X + theta_opt[0])
    plot_data(X, y)