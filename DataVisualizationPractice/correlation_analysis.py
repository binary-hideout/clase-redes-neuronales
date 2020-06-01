# Este programa carga despliega un analisis de correlaciones entre atributos de una base de datos
import pandas
import numpy as np
import matplotlib.pyplot as plt

# nombres de los atributos
names = [
    ' criminalidad ', ' lotes por zona ', ' porcentaje de negocios ', ' rio ',
    ' conc. oxidos ', ' # habitaciones ', ' edad de propiedad ', ' distancia a ctros. empleo ', 
    ' acceso a vias ', ' predial ', ' razon maestros por alumno', ' tipo de poblacion ',
    ' porcentaje pob. bajo ingreso ', 'valor de propiedad en miles de usd'
]
# números de los atributos correspondientes al orden de arriba
names2 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']

# leer contenido de CSV
data = pandas.read_csv('boston.csv', names=names)
# redondea los valores a 3 decimales
pandas.set_option('precision', 3)
# calcula la correlación del contenido
correlations = data.corr(method= 'pearson')
# imprimir correlaciones en pantalla
print(correlations)
# guardar correlaciones en archivo
np.savetxt('correlations.csv', correlations, delimiter=',')

# graficar matriz de correlación
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, 14, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names2)
ax.set_yticklabels(names)
plt.suptitle('Correlaciones', fontsize=14)
plt.show()
