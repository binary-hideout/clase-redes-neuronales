'''
Modelo de entrenamiento para una red neuronal de clasificación para el dataset del sonar.
'''

import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# leer datos de entrenamiento
x_train = np.loadtxt('xsonarTrain.csv', delimiter=',')
y_train = np.loadtxt('ysonarTrain.csv', dtype=str)
# las salidas deben ser binarias
y_train = LabelEncoder().fit_transform(y_train)

# modelo de red neuronal con 3 capas ocultas
ann = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(90, 60, 30), random_state=1)
# entrenar red
ann.fit(x_train, y_train)
# guardar red en archivo
with open('trainedmodel_sonar.sav', 'wb') as file:
    pickle.dump(ann, file)
