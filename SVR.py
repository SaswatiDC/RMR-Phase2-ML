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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


df=pd.read_csv('Combined values.csv')
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
scaler_x = StandardScaler()

#scaler_y = StandardScaler()
x = scaler_x.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



# Define the SVR model
svr = SVR(kernel='rbf', C=100, epsilon=0.2)


svr.fit(x_train, y_train)


y_pred = svr.predict(x_test)

model=SVR()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

combined_df=pd.DataFrame ({'y_test':y_test,'y_pred':y_pred})

sorted_df=combined_df.sort_values(by='y_test')
print(sorted_df)

print(f"Mean Squared Error: {mse}")
print(f"R-squared (R2) Score: {r2}")

coefficients = svr.coef_
intercept = svr.intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)







