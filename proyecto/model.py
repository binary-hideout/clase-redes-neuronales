import pickle

from pandas import read_csv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# leer contenido de base de datos
dataset = read_csv('Churn_Modelling.csv')
# asignar columnas 3-12 a entradas
x = dataset.iloc[:, 3:13].values
# asginar la última columna a la salida
y = dataset.iloc[:, -1].values

# la columna de países se debe transformar en números
x[:, 1] = LabelEncoder().fit_transform(x[:, 1])
# los géneros deben estar en números
x[:, 2] = LabelEncoder().fit_transform(x[:, 2])
# si se usan los números de los países tal como están (0,1,2),
# la red neuronal les asignará una prioridad a los de mayor número,
# pero como son categorías esto no es correcto,
# por lo que la columna se transforma en 2 y cada país
# será representado como número binario
x = ColumnTransformer([('', OneHotEncoder(), [1])], 'passthrough').fit_transform(x)[:, 1:]

# estandarización de los datos,
# para que estén en la misma escala numérica
scaler = StandardScaler().fit(x)
x = scaler.transform(x)

# separar datos en entrenamiento y en prueba
x_train, x_test, y_train, y_test = train_test_split(x, y)

# intentar cargar red de archivo
try:
    # abrir archivo binario
    with open('trained_ann.sav', 'rb') as file:
        # cargar contenido
        classifier = pickle.load(file)
# si no existe un modelo ya entrenado
except FileNotFoundError:
    # se inicializa una red neuronal sin capas
    classifier = Sequential()
    # agregar 3 capas ocultas
    for i in range(3):
        # 6 nodos, es el promedio de las entradas con las salidas, (12+1)/2
        # la función de activación es lineal exponencial (ELU)
        # los pesos son generados aleatoriamente
        hidden_layer = Dense(6, activation='elu', kernel_initializer='uniform')
        classifier.add(hidden_layer)
    # agregar capa de salida
    # 1 nodo, pues es salida binaria
    # función de activación sigmoide porque la salida será una probabilidad
    output_layer = Dense(1, activation='sigmoid', kernel_initializer='uniform')
    classifier.add(output_layer)

    # parámetros para crear la red neuronal
    # función de pérdida binaria porque la salida es binaria
    # la evaluación se basará en la métrica de precisión
    classifier.compile('nadam', loss='binary_crossentropy', metrics=['accuracy'])
    # entrenar red neuronal con dataset de entrenamiento
    classifier.fit(x_train, y_train, batch_size=10, epochs=30)

    # guardar red entrenada en archivo binario
    with open('trained_ann.sav', 'wb') as file:
        pickle.dump(classifier, file)

# salidas predichas como probabilidades
y_pred = classifier.predict(x_test)
# salidas predichas como binarias
y_pred_binary = (y_pred > 0.5)
# matriz de confusión para visualizar precisión
matrix = confusion_matrix(y_test, y_pred_binary)
print('Confusion matrix:\n', matrix)
# error de precisión
error = 1 - accuracy_score(y_test, y_pred_binary)
print(f'Error: {error:g}')

print()
# ID: 15628319
# Apellido: Bonaparte
# Crédito: 600
# Nacionalidad: Francia
# Género: Hombre
# Edad: 40
# Antigüedad: 3
# Balance: 60000
# Productos: 2
# Tarjeta de crédito: Sí
# Miembro activo: Sí
# Salario estimado: 50000
new_observation = [[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]
scaled_observation = scaler.transform(new_observation)
predict_observation = classifier.predict(scaled_observation)
binary_observation = (predict_observation > 0.5)
print('Probabilidad:', predict_observation)
print('Predicción binaria:', binary_observation)
