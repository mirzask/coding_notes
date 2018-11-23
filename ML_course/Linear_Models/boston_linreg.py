import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# Load data

from sklearn.datasets import load_boston

boston = load_boston()

boston.keys()

print(boston['DESCR'])

X = boston.data
y = boston.target


# Train Test Split time
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=101)

# Fit model

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)

# Calculate predictions

predictions = lm.predict(X_test)
predictions[:10]

# Evaluate

sns.regplot(y_test, predictions)



from sklearn.metrics import mean_squared_error

#MSE
mean_squared_error(y_true=y_test, y_pred=predictions)


#RMSE
np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions))

#R^2
from sklearn.metrics import r2_score

r2_score(y_test, predictions)
