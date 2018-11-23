import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Import data
df = pd.read_csv('USA_Housing.csv')

# Take a peek
df.head()

df.info()

df.describe()

df.columns

sns.pairplot(df)

sns.distplot(df['Price'])

sns.heatmap(df.corr(), annot=True)


# Step 1
# Create training and test sets

df.columns

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]

y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=101)

# Step 2
# Fit model

lm = LinearRegression()

lm.fit(X_train, y_train)
# Get intercept
lm.intercept_

# Get coefficients
lm.coef_


X_train.columns

# Create dataframe of coefficiencts

coeff_df = pd.DataFrame(lm.coef_, X_train.columns, columns=['Coeff'])
coeff_df

# Step 3
# Calculate predicted output (y)

predictions = lm.predict(X_test)
predictions

# Step 4
# Compare predicted results to actual results from "test" set

### Plots

plt.scatter(predictions, y_test)

sns.regplot(predictions, y_test)

## Histogram of residuals

sns.distplot((y_test - predictions))

#### Nice to see that the residuals are normally distributed

## Metrics: MAE, MSE, RMSE

### MAE (units)

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_true=y_test, y_pred=predictions)

### MSE (units^2)

from sklearn.metrics import mean_squared_error

mean_squared_error(y_true=y_test, y_pred=predictions)

### RMSE (units)
#from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions))

### R-squared

from sklearn.metrics import r2_score

r2_score(y_true=y_test, y_pred=predictions)
