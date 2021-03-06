import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

wine = pd.read_csv('https://raw.githubusercontent.com/hadley/rminds/master/1-data/wine.csv')

wine.dtypes # check to see if all features are numeric

# Drop non-numeric feature(s)
wine.drop('type', axis = 1, inplace=True)

X = wine.drop(['proline'], axis=1)

y = wine['proline']

X_train, X_test, y_train, y_test = train_test_split(X, y)

# KNN *without* normalization

from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor()
knn.fit(X_train, y_train)

knn.score(X_test, y_test) # meh, not that great



# Definition: Standardization
# Convert continuous, numerical (i.e. non-categorical) data to make it ~ normally distributed

# Additional methods: https://scikit-learn.org/stable/modules/preprocessing.html


############## LOG-NORMALIZATION ######################

# Useful if high variance, e.g. 'Proline' in wine dataset
# Performs log-transformation


import numpy as np

# Calculate variance of each feature
wine.apply(np.var) # proline has crazy high variance

# Normalize 'Proline'
wine['proline'] = np.log(wine['proline'])
np.var(wine['proline'])


###################### SCALING #########################

# StandardScaler - just turns things into Z-scores
## Z = x - mu / sd

# Useful if continuous features on diff scales, but
# operating on linear scale, e.g. KNN, lin reg


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

wine_subset = wine[['ash', 'alcalinity', 'magnesium']]

wine_subset.apply(np.var) # see how variance varies

wine_subset_scaled = scaler.fit_transform(wine_subset)

np.var(wine_subset_scaled) # so long variation in variance


# Standardizing using `mlxtend`
## Also creates a Z-score, but easier when working w/ Pandas DF

from mlxtend.preprocessing import standardize

standardize(wine, columns=['ash', 'alcalinity', 'magnesium'])

standardize(wine_subset)





# MinMaxScaler
## X_norm = (X - X_min) / (X_max - X_min)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

wine_minmax_scaled = scaler.fit_transform(wine_subset)

np.var(wine_minmax_scaled)


# Using mlxtend version of MinMaxScaler

from mlxtend.preprocessing import minmax_scaling

minmax_scaling(wine, columns=['ash', 'alcalinity', 'magnesium'])





########### Repeat KNN *with* scaling #################

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)


knn = KNeighborsRegressor()
knn.fit(X_train, y_train)

knn.score(X_test, y_test)
