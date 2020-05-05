'''
Este programa carga despliega un analisis de correlaciones entre atributos de una base de datos
'''

import pandas
import numpy as np
import matplotlib.pyplot as plt

filename = np.array(['renewable_energy.csv', 'petroleum.csv', 'electric_cars.csv'])

i = 1
for f in filename:
    series = pandas.read_csv(f)
    series.plot()

print(series)
plt.show()
