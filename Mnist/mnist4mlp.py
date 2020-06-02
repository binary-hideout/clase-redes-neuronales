#Este programa prueba conjuntamente (para fines comparativos) una red neuronal de 3 capas 
#y un regresor logistico pre-entrenados para identificacion de digitos MNIST. 
#Se cargan imagenes al azar de MNIST_TEST, se despliegan visualmente y el regresor logistico trata de identificar la imagen

import pickle

import numpy as np  
import matplotlib.pyplot as plt

# leer base de datos CSV
data = np.loadtxt('mnist_test.csv', delimiter=',')
ncol = data.shape[1]
# definiendo entradas y salidas
X_test = data[:, 1:ncol]
y_test = data[:, 0]

# se carga el regresor logistico entrenado
regression = pickle.load(open('finalized_model.sav', 'rb'), encoding='latin1')
# se carga la red neuronal entrenada
ann = pickle.load(open('finalized_modelMLP.sav', 'rb'), encoding='latin1')
# se escoge al azar un indice de imagen de prueba
ex = np.random.randint(0, 10000)

# el regresor logistico trata de identificar la imagen
xtest = X_test[ex,].reshape(1, -1)
predicted = regression.predict(xtest)
print("La regresion logistica predice un:")
print(predicted)

# la red neuronal trata de identificar la imagen
xtest = X_test[ex,].reshape(1, -1)
predicted = ann.predict(xtest)
print("La red neuronal predice un:")
print(predicted)

# la imagen realmente es
real = y_test[ex]
print("La imagen realmente es un:")
print(int(real))

# se genera una matriz de 28x28 para guardar imagen en escala de grises
select_image = np.random.random([28, 28])
# se traduce el vector seleccionado a imagen y se despliega visualmente
select_image1 = 1 - X_test[ex,] / 255
k = 0
for i in range(0, 28):
    for j in range(0, 28):
        select_image[i, j] = select_image1[k]
        k = k + 1
plt.imshow(select_image, cmap='gray', interpolation='nearest')
plt.show()
