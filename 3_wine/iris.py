from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# добавить цветовое разделение по классам (у каждого цвет свой на графиках)

cancer = load_wine() #длина, ширина лепестка


data = pd.DataFrame(cancer.data, columns=cancer.feature_names) # входные атрибуты (признаки), которые там что-то в data_frame
target = pd.DataFrame(cancer.target, columns=["target"]) # размерность 150 на 4 столбца
df = pd.concat([data, target], axis=1)
print(df)


# тренировочные данные train(70%), test (30%)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()                   # функция которая нормирует границы от 0 до 1
X_train = scaler.fit_transform(X_train)     # разбили X_train на 30% и 70% (получилось 105 строк на 4 столбца)
print(X_train)
X_test = scaler.fit_transform(X_test)
print(X_test)


# метод классификации
# У y_train есть метка класса, у тестовой метки нет

knn = KNeighborsClassifier(n_neighbors = 4) # метод k-ближайших соседей с определенными параметрами (число от 1 до 7)
knn.fit(X_train, y_train)   # обучаем модель
print(y_train)

y_pred = knn.predict(X_test) # прогноз (через метод predict мы подаем тестовую выборку)

print("Accuracy:", accuracy_score(y_test, y_pred)) # сумма верноклассифицированных точек / на общее кол-во точек

# корректная формула - посчитать сначала отдельно по классам

species = []

# добавляем названия
for i in range(len(df['target'])):
    if df['target'][i] == 0:
        species.append("setosa")
    elif df['target'][i] == 1:
        species.append('versicolor')
    else:
        species.append('virginica')

df['species'] = species

# разбиение по цветам
setosa = df[df.species == 'setosa']
versicolor = df[df.species == 'versicolor']
virginica = df[df.species == 'virginica']

fig, ax = plt.subplots()
fig.set_size_inches(13, 7) # задаём размеры графика.


# подписываем и отрисовываем точки.
ax.scatter(setosa['alcohol'], setosa['color_intensity'], label="class-0", facecolor="blue")
ax.scatter(versicolor['alcohol'], versicolor['color_intensity'], label="class-1", facecolor="green")
ax.scatter(virginica['alcohol'], virginica['color_intensity'], label="class-2", facecolor="red")

ax.set_xlabel("Алкоголь")
ax.set_ylabel("Интенсивность цвета")
ax.grid()
ax.set_title("Вино")
ax.legend()
plt.show()

print(y)