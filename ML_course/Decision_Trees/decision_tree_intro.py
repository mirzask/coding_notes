from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

reg1 = DecisionTreeRegressor(max_depth=1)
reg2 = DecisionTreeRegressor(max_depth=2)

X = np.linspace(-2, 2, 7).reshape(-1, 1)
y = X ** 3

# Fit the model
reg1.fit(X, y)
reg2.fit(X, y)

# Predictions
X_test = np.arange(-2.0, 2.0, 0.1)[:, np.newaxis]
y_1 = reg1.predict(X_test)
y_2 = reg2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=1", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=2", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()



