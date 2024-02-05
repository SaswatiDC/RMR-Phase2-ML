from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# Create different base regressors
base_regressor_1 = DecisionTreeRegressor()
base_regressor_2 = LinearRegression()

# Assuming you have your data in X and y
df=pd.read_csv('Combined values.csv')
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_regressor = KNeighborsRegressor(n_neighbors=3)
# Create custom ensemble by combining the predictions of both base regressors
##def custom_ensemble_predict(X, base_regressor_1, base_regressor_2):
##    predictions_1 = base_regressor_1.predict(X)
##    predictions_2 = base_regressor_2.predict(X)
##    # You can combine the predictions in a way that makes sense for your problem, for example, averaging them
##    ensemble_predictions = (predictions_1 + predictions_2) / 2
##    return ensemble_predictions

# Create a Bagging Regressor with the custom ensemble
bagging_regressor = BaggingRegressor(base_estimator=base_regressor_1, n_estimators=5, random_state=42, bootstrap=True)

# Fit the Bagging Regressor
bagging_regressor.fit(X_train, y_train)
y_pred=bagging_regressor.predict(X_test)

# Make predictions using the custom ensemble
# ensemble_predictions = custom_ensemble_predict(X, base_regressor_1, base_regressor_2)

# Now, ensemble_predictions contains the combined predictions from both base regressors


combined_df=pd.DataFrame ({'y_test':y_test,'y_pred':y_pred})

sorted_df=combined_df.sort_values(by='y_test')
print(sorted_df)
               

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared (R2) Score: {r2}")
