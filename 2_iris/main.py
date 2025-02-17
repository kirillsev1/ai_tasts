import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


df=pd.read_csv('iris.csv', delimiter = ',')
print(df)
print(df.shape)

cancer = load_iris()
print(cancer)

data = cancer.data

# Извлечение первого и второго чисел из вложенных списков
x1 = [item[0] for item in data]
x2 = [item[1] for item in data]
x3 = [item[2] for item in data]
x4 = [item[3] for item in data]


plt.scatter(x1, x2)
plt.title('Зависимость первого числа от второго')
plt.xlabel('Первое число')
plt.ylabel('Второе число')
plt.grid()
plt.show()

plt.scatter(x1, x3)
plt.title('Зависимость первого числа от третьего')
plt.xlabel('Первое число')
plt.ylabel('Третье число')
plt.grid()
plt.show()

plt.scatter(x1, x4)
plt.title('Зависимость первого числа от четвертого')
plt.xlabel('Первое число')
plt.ylabel('Четвертое число')
plt.grid()
plt.show()

