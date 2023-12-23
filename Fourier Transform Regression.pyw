from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)
df=pd.read_csv('Combined values.csv')
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Function to add Fourier features
def add_fourier_features(X, num_terms):
    n_records, n_inputs = X.shape
    X_new = np.zeros((n_records, n_inputs * (2 * num_terms + 1)))
    
    for i in range(num_terms):
        for j in range(n_inputs):
            X_new[:, i * n_inputs + j] = np.sin((i + 1) * X[:, j])
            X_new[:, (i + num_terms) * n_inputs + j] = np.cos((i + 1) * X[:, j])
    
    return X_new

# Add Fourier features to the input data.
# Final featues are x1, x2, x3, x4, sinx1,cosx1, sin2x1, cos2x1...,sinx2,cosx2, sin2x2, cos2x2...,...,cos2x4,...
K = 107 # number of Fourier terms for each feature including itsef
X_train_transformed = add_fourier_features(x_train,K)
X_test_transformed = add_fourier_features(x_test,K)
# Create a pipeline with standard scaling and linear regression
model = make_pipeline(StandardScaler(), LinearRegression())

# Fit the model
model.fit(X_train_transformed, y_train)
y_pred = model.predict(X_test_transformed)

print("Predictions:", y_pred)

mse = mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse}')
print(f'{r2}')
print(model.named_steps['linearregression'].coef_)
print(model.named_steps['linearregression'].intercept_)
combined_df=pd.DataFrame ({'y_test':y_test,'y_pred':y_pred})

sorted_df=combined_df.sort_values(by='y_test')


### Optional: Visualize the results
##plt.scatter(y, model.predict(X_new), label='Training Data')
##plt.scatter(y_pred, y_pred, color='red', marker='x', label='Test Predictions')
##plt.xlabel('Actual Values')
##plt.ylabel('Predicted Values')
##plt.legend()
##plt.show()
