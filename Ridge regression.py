import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Generate some random data
df=pd.read_csv('Combined values.csv')
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a Ridge regression model
alpha = 0.1  # regularization strength
ridge_reg = Ridge(alpha=alpha)
ridge_reg.fit(x_train, y_train)

# Make predictions on the test set
y_pred = ridge_reg.predict(x_test)

combined_df=pd.DataFrame ({'y_test':y_test,'y_pred':y_pred})

sorted_df=combined_df.sort_values(by='y_test')
print(sorted_df)

# Print the coefficients
print("Coefficients:", ridge_reg.coef_)
print( ridge_reg.intercept_)



# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse}')
print(f'{r2}')
