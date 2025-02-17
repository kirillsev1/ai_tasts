from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


wine = load_wine()

X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.DataFrame(wine.target, columns=['quality'])

df = pd.concat([X, y], axis=1)

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# СТАНДАРТИЗИРУЕМ
scaler = StandardScaler()   # функция которая нормирует границы от 0 до 1
X_train = scaler.fit_transform(X_train)  # разбили X_train на 30% и 70% (получилось 105 строк на 4 столбца)
X_test = scaler.fit_transform(X_test)

# ЛИНЕЙНАЯ РЕГРЕССИЯ
lin_reg = LinearRegression()    # сформировали модель # у линейнлй регрессии есть различные параметры
lin_reg.fit(X_train, y_train)    # метод линейной модели позволяет обучать данные, которые подали как аргумент, он запоминает, у каких точек какие метки класса должны быть
y_pred_lin = lin_reg.predict(X_test)
print(y_pred_lin)

mse_linear = mean_squared_error(y_test, y_pred_lin)
print(mse_linear)
r2_linear = r2_score(y_test, y_pred_lin)
print(f"Лучший MSE (Linear): {mse_linear}")
print(f"Лучший R^2 (Linear): {r2_linear}")

# RIDGE РЕГРЕССИЯ
ridge_reg = Ridge()
param_grid_ridge = {'alpha': np.logspace(-3, 3, 7)}
# запускаем алгоритм кросс-валидации (1 прогон выборки = 5 валидаций (4 в составе обучайщей, 1 раз в тестовой))
grid_search_ridge = GridSearchCV(ridge_reg, 
                                param_grid_ridge, cv=5, 
                                scoring='neg_mean_squared_error') #
result = grid_search_ridge.fit(X_train, y_train)
# print(result)

# было 5 запусков, покажется наилучшее значение
print('Лучшие параметры гребневой регрессии:', grid_search_ridge.best_params_)
print('Лучшая оценка на кросс-валидации (MSE) (Ridge):', grid_search_ridge.best_score_)

# best_estimator выводит наилучшую модель
# какая модель наиболее эффективно для решения задачи
best_ridge_regressor = grid_search_ridge.best_estimator_
# print(best_ridge_regressor)
# применяем predict и для наилучшей обученной модели подаем тестовую выборку
y_pred_ridge = best_ridge_regressor.predict(X_test)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"Лучший MSE (Ridge): {mse_ridge}")
print(f"Лучший R^2 (Ridge): {r2_ridge}")

# LASSO РЕГРЕССИЯ
lasso_regressor = Lasso()
param_grid_lasso = {'alpha': np.logspace(-3, 3, 7)}

grid_search_lasso = GridSearchCV(lasso_regressor, 
                                param_grid_lasso,
                                scoring='neg_mean_squared_error')
grid_search_lasso.fit(X_train, y_train)

best_lasso_regressor = grid_search_lasso.best_estimator_
y_pred_lasso = best_lasso_regressor.predict(X_test)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f"Лучший MSE (Lasso): {mse_lasso}")
print(f"Лучший R^2 (Lasso): {r2_lasso}")

# Plot Results
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression'],
    'MSE': [mean_squared_error(y_test, y_pred_lin),
            mean_squared_error(y_test, y_pred_ridge),
            mean_squared_error(y_test, y_pred_lasso)],
    'R-squared': [r2_score(y_test, y_pred_lin),
                 r2_score(y_test, y_pred_ridge),
                 r2_score(y_test, y_pred_lasso)]
})

if r2_linear > r2_ridge and r2_linear > r2_lasso:
    print(
        "ИТОГ: Линейная регрессия показала наилучшие результаты и лучше всего подходит для данной задачи."
    )
elif r2_ridge > r2_linear and r2_ridge > r2_lasso:
    print(
        "ИТОГ: Гребневая регрессия показала наилучшие результаты и лучше всего подходит для данной задачи."
    )
elif r2_lasso > r2_linear and r2_lasso > r2_ridge:
    print(
        "ИТОГ: Лассо-регрессия показала наилучшие результаты и лучше всего подходит для данной задачи."
    )

# plt.figure(figsize=(10, 6))
# plt.bar(results['Model'], results['MSE'], color=['blue', 'orange', 'green'])
# plt.title('Comparison of Regression Models')
# plt.xlabel('Model')
# plt.ylabel('Mean Squared Error')
# plt.show()

print("\nComparison of Regression Models:")
print(results)

plt.figure(figsize=(10, 6))
plt.bar(results['Model'], results['R-squared'], color=['blue', 'orange', 'green'])
plt.title('Comparison of Regression Models')
plt.xlabel('Model')
plt.ylabel('Mean Squared Error')
plt.show()