import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
# Load the data
data = pd.read_csv('Food Delivery Time Prediction Case Study.csv')

# Reshape the data because sklearn expects the inputs to be 2D arrays
X_simple = np.array(data['Delivery_person_Ratings']).reshape(-1, 1)
y = data['Time_taken(min)']

# Create a linear regression model and fit it to the data
model_simple = LinearRegression()
model_simple.fit(X_simple, y)

# Predict the response variable
y_pred_simple = model_simple.predict(X_simple)

# Calculate the R-squared score and root mean squared error (RMSE)
r2_simple = r2_score(y, y_pred_simple)
multiple_R = np.sqrt(r2_simple)
rmse_simple = np.sqrt(mean_squared_error(y, y_pred_simple))
# Coefficients
coefficients = model_simple.coef_

# Intercept
intercept = model_simple.intercept_

# calculate residuals
residuals = y - y_pred_simple

# Sum of Squares Total (SST)
SST = sum((y - np.mean(y))**2)

# Sum of Squares Regression (SSR)
SSR = sum((y_pred_simple - np.mean(y))**2)

# Sum of Squares Error (SSE)
SSE = sum((y - y_pred_simple)**2)  # or sum(residuals**2)


# Multiple Linear Regression
# Create a 2D array of the predictors
X_multiple = data[['Delivery_person_Ratings', 'Delivery_person_Age']]

# Create a linear regression model and fit it to the data
model_multiple = LinearRegression()
model_multiple.fit(X_multiple, y)

# Predict the response variable
y_pred_multiple = model_multiple.predict(X_multiple)

# Calculate the R-squared score and root mean squared error (RMSE)
r2_multiple = r2_score(y, y_pred_multiple)
rmse_multiple = np.sqrt(mean_squared_error(y, y_pred_multiple))

# Print the results
print('Simple linear regression')
print('R-squared score: {:.3f}'.format(r2_simple))
print('Multiple R: {:.3f}'.format(multiple_R))
print('RMSE: {:.3f}'.format(rmse_simple))
print('y = {:.3f}x + {:.3f}'.format(coefficients[0], intercept))
print('SST: {:.3f}'.format(SST))
print('SSR: {:.3f}'.format(SSR))
print('SSE: {:.3f}'.format(SSE))
print('\nMultiple linear regression')
print('R-squared score: {:.3f}'.format(r2_multiple))
print('RMSE: {:.3f}'.format(rmse_multiple))

## Scatter plot with line of best fit
plt.figure(figsize=(10, 6))
sns.regplot(x='Delivery_person_Ratings', y='Time_taken(min)', data=data, line_kws={"color": "red"})
plt.title('Rating vs Time taken', fontsize=20)
plt.xlabel('Rating', fontsize=15)
plt.ylabel('Time taken (min)', fontsize=15)
plt.tight_layout()
plt.savefig('graphs/linear_regression.png')
plt.show()

