from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

# leyendo el csv sacando los datos de X y Y
x = np.loadtxt('xbostonTrain.csv', delimiter=',', usecols=range(13))
y = np.loadtxt('ybostonTrain.csv', delimiter=',', usecols=(0, ))

#  se construyen conjuntos de entrenamiento y prueba al azar
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=5)

# se usa el conjunto de prueba para entrenar el clasificador
clf = MLPRegressor(solver='lbfgs', alpha=1e-5, tol=1e-100, hidden_layer_sizes=(15,), random_state=1, max_iter=500)
clf.fit(x_train, y_train)

# se usa el modelo entrenado para predecir las salidas sobre el conjunto de prueba
predicted = clf.predict(x_test)

# se calcula el error sobre el conjunto de prueba
error = mean_squared_error(y_test, predicted)
print('Error cuadrado medio sobre el conjunto de prueba:', error)
print('Raiz cuadrada del error:', np.sqrt(error))
print('Desviacion estandar de las salidas:', np.sqrt(np.var(y_train)))

# se guarda el modelo entrenado para uso posterior
filename = 'trained_modelMLP.sav'
pickle.dump(clf, open(filename, 'wb'))
