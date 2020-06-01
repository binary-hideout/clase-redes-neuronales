'''
Modelo de prueba para la red neuronal de clasificación para el dataset del sonar.
'''

import pickle

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# leer datos de prueba
x_test = np.loadtxt('xsonarTest.csv', delimiter=',')
y_test = np.loadtxt('ysonarTest.csv', dtype=str)
# las salidas deben ser binarias
y_test = LabelEncoder().fit_transform(y_test)

# cargar red de archivo
with open('trainedmodel_sonar.sav', 'rb') as file:
    ann = pickle.load(file)

# predecir salidas
predicted = ann.predict(x_test)

# error de predicción
error = 1 - accuracy_score(y_test, predicted)
print(f'Error de predicción: {error}')
