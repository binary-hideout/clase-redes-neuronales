'''
Modelo de entrenamiento para una red neuronal de regresión para el dataset de Boston.
'''

import pickle

from sklearn.neural_network import MLPRegressor
import numpy as np

# leer datos de entrenamiento
x_train = np.loadtxt('xbostonTrain.csv', delimiter=',')
y_train = np.loadtxt('ybostonTrain.csv')

# modelo de red neuronal con 4 capas ocultas
ann = MLPRegressor(solver='lbfgs', alpha=1e-5, tol=1e-100, hidden_layer_sizes=(28, 21, 14, 7), max_iter=10**4, random_state=1)
# entrenar red
ann.fit(x_train, y_train)
# guardar red en archivo
with open('trainedmodel_boston.sav', 'wb') as file:
    pickle.dump(ann, file)
