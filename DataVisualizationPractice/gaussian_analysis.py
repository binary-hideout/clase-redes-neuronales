# Este programa carga un conjunto de muestras de datos en formato csv, las estandariza,
# les aplica una prueba de Kolmogorov-Smirnov y despliega visualmente 
# los histogramas correspondientes
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# nombres de los archivos
filename = np.array(['Sample_33_data.txt', 'Sample_34_data.txt', 'Sample_35_data.txt', 'Sample_33_data_down.txt', 'Sample_34_data_down.txt', 'Sample_35_data_down.txt'])# 352Km to 359Km region

ydata = np.empty([1025, 6])
i = 1
for f in filename:
    print(f)
    # leer contenido de archivo
    x, y = np.loadtxt(fname=f, usecols=(0, 1), unpack=True)
    ym = np.mean(y)
    ys = np.std(y)
    y = (y - ym) / ys
    ks = stats.kstest(y, 'norm')
    # imprimir resultados en consola
    print(ks)

    # añadir histograma de archivo
    ax = plt.subplot(2, 3, i)
    i = i + 1
    ax.hist(y, bins=100)

# mostrar gráfica
plt.show()
