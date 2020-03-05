from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

# leyendo el csv sacando los datos de X y Y
x = np.loadtxt('irisTrain1.csv', delimiter=',', usecols=range(4))
y = np.loadtxt('irisTrain1.csv', delimiter=',', usecols=(4, ), dtype='str')
	
	# se construyen conjuntos de entrenamiento y prueba al azar
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=5)

# se usa el conjunto de prueba para entrenar el clasificador
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
clf.fit(x_train, y_train)

# se usa el modelo entrenado para predecir las salidas sobre el conjunto de prueba
predicted = clf.predict(x_test)

# se calcula el error sobre el conjunto de prueba
error = 1 - accuracy_score(y_test, predicted)
print('Error sobre el conjunto de prueba:', error)

# se guarda el modelo entrenado para uso posterior
filename = 'trained_modelMLP.sav'
pickle.dump(clf, open(filename, 'wb'))
