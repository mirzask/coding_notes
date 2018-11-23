from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



df = pd.read_csv('data/telecom_churn.csv')

df.head()


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


# Setup and fit model

tree = DecisionTreeClassifier(max_depth=5, random_state=17)

tree.fit(X_train, y_train)


# Generate predictions and Assess model accuracy

pred = tree.predict(X_test)

accuracy_score(y_test, pred)


# Generate decision tree dot file

import pydotplus
from sklearn.tree import export_graphviz

def tree_graph_to_png(tree, feature_names, png_file_to_save):
    tree_str = export_graphviz(tree, feature_names=feature_names,
                                     filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(png_file_to_save)


tree_graph_to_png(tree=tree, feature_names=df.columns, png_file_to_save='data/my_tree.png')

# Hyper-parameter tuning

tree_params = {'max_depth': range(1,11),
               'max_features': range(4,19)}

tree_grid = GridSearchCV(tree, tree_params,
                         cv=5, n_jobs=-1, verbose=True)

tree_grid.fit(X_train, y_train)




# Evaluate results of cross validation

tree_grid.best_params_
tree_grid.best_score_
tree_grid.best_estimator_

# Assess accuracy of "best" model from CV

pred_CV = tree_grid.predict(X_test)

accuracy_score(y_true=y_test, y_pred=pred_CV)


# Generate decision tree graph

tree_graph_to_png(tree=tree_grid, feature_names=df.columns, png_file_to_save='data/my_tree_CV.png')
