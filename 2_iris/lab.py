from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


# МЕТОД КЛАССИФИКАЦИИ
# + метод k-ближайших соседей с определенными параметрами

# вспомнить, что такое cancer 
cancer = load_iris()        # подгружаем load_iris # длина, ширина лепестка

# приводим к формату DataFrame из библиотеки pandas
data = pd.DataFrame(cancer.data, columns=cancer.feature_names)      # входные атрибуты (признаки), которые там что-то в data_frame
target = pd.DataFrame(cancer.target, columns=["target"])            # размерность 150 на 4 столбца
df = pd.concat([data, target], axis=1)
print(df)


# есть тренировочные данные train(на них приходится 70%) и test (на них 30%)
# работаем отдельно с тренировочными данными и с тестовыми, обучая модель

X = df.drop("target", axis=1)                   # drop отделяет первый столбец данных, для чего??(найти)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = StandardScaler()                       # функция которая нормирует границы от 0 до 1
X_train = scaler.fit_transform(X_train)         # разбили X_train на 30% и 70% (получилось 105 строк на 4 столбца)
print(X_train)
X_test = scaler.fit_transform(X_test)           # тестовые данные
print(X_test)


## метод классификации
# У train есть метка класса, у test метки нет (еще раз посмотреть, что за метка)
knn = KNeighborsClassifier(n_neighbors = 4)     # метод k-ближайших соседей с определенными параметрами (число от 1 до 7)
knn.fit(X_train, y_train)
print(y_train)


# проверяем, как модель обучилась - делаем предсказание метки класса (метод predict)
y_pred = knn.predict(X_test)                    # через метод predict мы подаем тестовую выборку


print("Accuracy:", accuracy_score(y_test, y_pred)) # сумма верноклассифицированных точек / на общее кол-во точек

# корректная полная формула - посчитать саначала отдельно по классам и потом уже применить формулу:
# сумма верноклассифицированных точек / на общее кол-во точек
