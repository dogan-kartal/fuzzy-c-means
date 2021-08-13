from pylab import *
from numpy import *
import pandas as pd
import numpy as np
import operator
import math
import matplotlib.pyplot as plt
import random
import time

# csv dosyasında bulunan gerçek verilerimizi okuma işlemleri
data_full = pd.read_csv("Iris.csv")
columns = list(data_full.columns)
features = columns[0:len(columns) - 1]
data = data_full[features]

# Küme sayısını belirliyoruz (setosa, versicolor, virginica)
c = 3
# Maksimum iterasyon sayısı
MAX_ITER = 100
# Epsilon değeri
Epsilon = 0.00000001
# Toplam verilerin sayısı
n = len(data)
# Fuzzy algoritmasında en iyi sonucu elde etmek için fuzzy parametresi değerini 2 olarak belirliyoruz
m = 2.00

# Matrisi rastgele değerlerle tanımlama
def initialize():

    U = list()

    for i in range(n):
        random_list = [random.random() for i in range(c)]
        summation = sum(random_list)
        temp_list = [x / summation for x in random_list]
        U.append(temp_list)
    return U


# Küme merkezinin hesaplanması işlemi (her iterasyonda tekrarlanır)
def calculateCenter(U):
    U_zhuanzhi = list(zip(*U))
    V = list()
    for j in range(c):
        x = U_zhuanzhi[j]
        xraised = [e ** m for e in x]
        denominator = sum(xraised)
        temp_num = list()
        for i in range(n):
            data_point = list(data.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z / denominator for z in numerator]
        print(center)
        V.append(center)
    return V


# Küme merkezleri hesaplandıktan sonra bunları kullanarak random oluşturulan matrisin güncellenmesi
def U_update(U, V):
    p = float(2 / (m - 1))
    for i in range(n):
        x = list(data.iloc[i])

        distances = [np.linalg.norm(list(map(operator.sub, x, V[j]))) for j in range(c)]
        for j in range(c):
            den = sum([math.pow(float(distances[j] / distances[c]), p) for c in range(c)])
            U[i][j] = float(1 / den)
    return U


# Iterasyon işlemlerinin yapıldığı fonksiyon
# Merkez noktaları hesaplanıp random matris güncellenir daha sonra bu güncellenen matrisin merkezi hesaplanır
# ve tekrar güncelleme işlemi yapılır
def iteration(U):
    iter = 0
    while iter <= MAX_ITER:
        iter += 1
        V = calculateCenter(U)
        U = U_update(U, V)
        V_update = calculateCenter(U)
        juli = 0
        for i in range(c):
            for j in range(len(columns) - 1):
                juli = (V_update[i][j] - V[i][j]) ** 2 + juli
        if (sqrt(juli) < Epsilon):
            break
    return V, U


# elde bulunan verinin hangi kümeye ait olduğunu belirleme
def getResult(U):
    results = list()
    for i in range(n):
        max_value, index = max((value, index) for (index, value) in enumerate(U[i]))
        results.append(index)
    return results


# FuzzyCMeans algoritmasının yapıldığı ana fonksiyon
def FCM():
    start = time.time()
    U = initialize()
    V, U = iteration(U)
    results = getResult(U)
    print("FuzzyCMeans algoritmasının zaman maliyeti：{0} s".format(time.time() - start))
    return results, V, U

results, V, U = FCM()
V_array = np.array(V)
DATA = np.array(data)
results = np.array(results)


# Verilerin birinci ve ikinci sütunlarını belirleme
xlim(4, 8)
ylim(1, 5)
# Çizim penceresi oluşturma
plt.figure(1)
# Scatter grafiği çizdirme
plt.scatter(DATA[nonzero(results == 0), 0], DATA[nonzero(results == 0), 1], marker='o', color='r', label='0', s=30)
plt.scatter(DATA[nonzero(results == 1), 0], DATA[nonzero(results == 1), 1], marker='+', color='b', label='1', s=30)
plt.scatter(DATA[nonzero(results == 2), 0], DATA[nonzero(results == 2), 1], marker='*', color='g', label='2', s=30)
# Kümelerin merkez noktalarını işaretleme
plt.scatter(V_array[:, 0], V_array[:, 1], marker='x', color='m', s=50)
plt.show()