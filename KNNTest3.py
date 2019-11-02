import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from KNearestRegressor import KNearestRegressors

data = pd.read_csv("Salary_Data.csv")
x = data.iloc[:, 0].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25)
knn = KNearestRegressors(k=11)
knn.fit(x_train=x_train, y_train=y_train)

y_pred = knn.predict(np.array(x_test).reshape(len(x_test), 1))
score = 0
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        score += 1
    print(y_test[i] + "->" + y_pred[i])

print(score, "/", len(y_test))



