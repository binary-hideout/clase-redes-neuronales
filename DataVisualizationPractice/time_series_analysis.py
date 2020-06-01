'''
Este programa carga despliega un analisis de correlaciones entre atributos de una base de datos
'''

# importar librerías

import pandas
import numpy as np
import matplotlib.pyplot as plt

# arreglo de archivos CSV a leer
filename = np.array(['renewable_energy.csv', 'petroleum.csv', 'electric_cars.csv'])

# para cada archivo CSV
for f in filename:
    # leer contenido
    series = pandas.read_csv(f)
    # imprimirlo en consola
    print(series)
    # graficar los atributos como x,y
    series.plot()

# mostrar las gráficas
plt.show()
