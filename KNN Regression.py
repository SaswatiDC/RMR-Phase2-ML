import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from matplotlib import cm
from sklearn.svm import SVR
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Combined values.csv')
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
#Normalize input features
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

parameter1_values = []
parameter2_values = []

for k in range (9):
    knn_regressor = KNeighborsRegressor(n_neighbors=k+1)
    knn_regressor.fit(x_train, y_train)
    y_pred = knn_regressor.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    parameter1_values.append(k+1)
    parameter2_values.append(mse)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared (R2) Score: {r2}")

    




combined_df=pd.DataFrame ({'y_test':y_test,'y_pred':y_pred})

sorted_df=combined_df.sort_values(by='y_test')
print(sorted_df)
plt.plot(parameter1_values, parameter2_values)
plt.title('Error vs. Number of Neighbors (K)')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Mean Squared Error (MSE)')
plt.show()

    

