'''
Modelo de prueba para la red neuronal de regresión para el dataset de Boston.
'''

import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# leer datos de prueba
x_test = np.loadtxt('xbostonTest.csv', delimiter=',')
y_test = np.loadtxt('ybostonTest.csv')

# cargar red de archivo
with open('trainedmodel_boston.sav', 'rb') as file:
    ann = pickle.load(file)

# predecir salidas
predicted = ann.predict(x_test)

# error de predicción
error = mean_squared_error(y_test, predicted)
print('Error cuadrado medio sobre el conjunto de prueba:', error)
print('Raiz cuadrada del error:', np.sqrt(error))
print('Desviacion estandar de las salidas:', np.sqrt(np.var(y_test)))

error = mean_absolute_error(y_test, predicted)
print(f'Error absoluto: {error}')
