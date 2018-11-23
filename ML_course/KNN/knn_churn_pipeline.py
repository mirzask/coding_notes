from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df = pd.read_csv('data/telecom_churn.csv')

df.head()


# Convert to numerics

# Convert Yes/No to 1/0
## pd.factorize gives output of array and index, hence [0] to take the array only

df['International plan'] = pd.factorize(df['International plan'], sort = True)[0]

df['Voice mail plan'] = pd.factorize(df['Voice mail plan'], sort = True)[0]

# Convert boolean (T/F) to integer

df['Churn'] = df['Churn'].astype('int')

df.dtypes # `State` is the only non-numeric variable now

# Remove states, but store it just in case

states = df.pop('State')


# Create X and y

y = df.pop('Churn')

X = df




# Build train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)




######### PIPELINE ############

# Incorporate the scaling into the pipeline

from sklearn.pipeline import Pipeline

knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])

knn_params = {'knn__n_neighbors': range(1, 11)}

knn_grid = GridSearchCV(knn_pipe, knn_params,
                        cv=5, n_jobs=-1, verbose=True)

knn_grid.fit(X_train, y_train)

knn_grid.best_params_, knn_grid.best_score_


accuracy_score(y_test, knn_grid.predict(X_test))





# How does accuracy change w/ `n_neighbors`?

import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline

cv_scores, holdout_scores = [], []
n_neighb = [1, 2, 3, 5] + list(range(50, 550, 50))

for k in n_neighb:
    knn_pipe = Pipeline([('scaler', StandardScaler()),
                     ('knn', KNeighborsClassifier(n_neighbors=k))])
    cv_scores.append(np.mean(cross_val_score(knn_pipe, X_train, y_train, cv=5)))
    knn_pipe.fit(X_train, y_train)
    holdout_scores.append(accuracy_score(y_test, knn_pipe.predict(X_test)))

plt.plot(n_neighb, cv_scores, label='CV'); plt.show()
plt.plot(n_neighb, holdout_scores, label='holdout')
plt.title('Accuracy of kNN by `n_neighbors`')
plt.legend();