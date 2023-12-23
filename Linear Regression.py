import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from matplotlib import cm
from sklearn.svm import SVR
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


df=pd.read_csv('Combined values.csv')
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)






model = LinearRegression()
model.fit(x_train, y_train)
y_pred=model.predict(x_test)


combined_df=pd.DataFrame ({'y_test':y_test,'y_pred':y_pred})

sorted_df=combined_df.sort_values(by='y_test')
print(sorted_df)
               


mse=((y_test-y_pred)**2).mean()

print(f"Mean Squared Error: {mse}")
print(r2_score(y_test,y_pred))
print( model.coef_)
print( model.intercept_)


##plt.plot(sorted_df['y_test'],sorted_df['y_pred'])
##
##
##plt.xlabel('x axis')
##plt.ylabel('y axis')
##plt.title('graph between y test and predicted')
##plt.legend()
##plt.show()




