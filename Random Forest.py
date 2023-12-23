from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
df=pd.read_csv('Combined values.csv')
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
# Create different base regressors
base_regressor_1 = DecisionTreeRegressor()
base_regressor_2 = LinearRegression()
base_regressor_3 =KNeighborsRegressor(n_neighbors=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Assuming you have your data in X and y

# Create a Random Forest Regressor with the default base estimator (DecisionTreeRegressor)
model = RandomForestRegressor(n_estimators=5, random_state=42)

# Create a Bagging Regressor with the custom ensemble
#model = BaggingRegressor(base_estimator=base_regressor_1, n_estimators=5, random_state=42, bootstrap=True)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

combined_df=pd.DataFrame ({'y_test':y_test,'y_pred':y_pred})

sorted_df=combined_df.sort_values(by='y_test')
print(sorted_df)
               
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared (R2) Score: {r2}")
