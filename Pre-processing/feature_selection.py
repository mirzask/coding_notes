import numpy as np
import pandas as pd

# Remove unchanged/low variance features
## Low variance features contain little/no add'l information

from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer()['data'], load_breast_cancer()['target']

X = pd.DataFrame(X, columns = load_breast_cancer()['feature_names'])
X.shape


# Variance of each feature
np.var(X)


# Set threshold as 0.2
thresholder = VarianceThreshold(threshold=0.2)

X_high_var = thresholder.fit_transform(X)
X_high_var.shape # went from 30 features to 11 features

## What features remain?
thresholder.get_support()

X.columns[thresholder.get_support()]

### As a df

X[X.columns[thresholder.get_support()]].head()



# Try w/ a threshold as 0.02
X_try2 = VarianceThreshold(threshold=0.02).fit_transform(X)
X_try2.shape # now we wind up with 14 features





# KBest and F-stat/Chi-squared
## Features that are "independent" of the target variable are likely unimportant
## If numeric -> use f_classif (ANOVA F-value)
## If categorical -> use chi2

from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# I set k = 10 to pick the 10 "best" features

feats = SelectKBest(f_classif, k=10)
feats.fit_transform(X, y)

feats.get_support()
X.columns[feats.get_support()]





############ SEQUENTIAL FEATURE SELECTION #################

#Includes Forward Selection vs Backward selection
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=4, n_jobs=-1))])


# Forward Selection

selector = SequentialFeatureSelector(knn_pipe, scoring='accuracy', forward=True,
                                     floating=False, k_features=3,
                                     verbose=2, n_jobs=-1, cv=5)

selector.fit(X=X, y=y)


selector.subsets_

selector.k_feature_idx_
selector.k_feature_names_
selector.k_score_


pd.DataFrame.from_dict(selector.get_metric_dict()).T




# Backward Selection
select_back = SequentialFeatureSelector(knn_pipe, k_features=3, forward=False,
                                        floating=False, verbose=2, scoring='accuracy',
                                        cv=5, n_jobs=1)


select_back.fit(X=X, y=y)



# Plot results of Feature Selection (using `mlxtend`)

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt

fig1 = plot_sfs(selector.get_metric_dict(), kind='std_dev')
plt.ylim([0.8, 1])
plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show();
