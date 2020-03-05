from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

# leyendo el csv sacando los datos de X y Y
x = np.loadtxt('irisTest1.csv', delimiter=',', usecols=range(4))
y = np.loadtxt('irisTest1.csv', delimiter=',', usecols=(4, ), dtype='str')

# se carga la red neuronal entrenada
clf = pickle.load(open('trained_modelMLP.sav', 'rb'))

# se usa el modelo entrenado para predecir las salidas 
predicted = clf.predict(x)

# se calcula el error de prediccion
error = 1 - accuracy_score(y, predicted)
print('Error de prediccion:', error)
