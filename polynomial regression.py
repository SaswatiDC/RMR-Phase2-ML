import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# Generate some random data
df=pd.read_csv('Combined values.csv')
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Create polynomial features
degree = 4  # degree of the polynomial
#poly_features = PolynomialFeatures(degree=degree, include_bias=True)
model = make_pipeline(PolynomialFeatures(degree),LinearRegression())
#x_train = poly_features.fit_transform(x_train)

# Train a linear regression model with polynomial features
##lin_reg = LinearRegression()
##lin_reg.fit(x_train, y_train)
##


# Make predictions on the test set
##x_range = np.linspace(0, 2, 100).reshape(-1, 1)
##x_range_poly = poly_features.transform(x_range)
##y_pred = lin_reg.predict(x_range_poly)



##
##coefficients=lin_reg_steps['linearregression'].coef_
##intercept=lin_reg_steps['linearregression'].intercept_


### Print the coefficients
##print("Coefficients:", coefficients)
##print( "intercept",intercept)

model.fit(x_train, y_train)
y_pred=model.predict(x_test)
print(model.named_steps['linearregression'].coef_)
print(model.named_steps['linearregression'].intercept_)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse}')
print(f'{r2}')
combined_df=pd.DataFrame ({'y_test':y_test,'y_pred':y_pred})
sorted_df=combined_df.sort_values(by='y_test')
print(sorted_df)
