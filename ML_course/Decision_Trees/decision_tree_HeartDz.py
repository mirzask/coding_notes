# Predicting Heart Disease

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('data/mlbootcamp5_train.csv',
                 index_col='id', sep=';')

df.head()

# Convert `age` to be in years

df['age'] = round(df['age'] / 365)


# One-Hot encoding for `cholesterol` and `gluc`

## Unique values for each

df.gluc.unique()

df.cholesterol.unique()

## Using get_dummies from pandas [umm... should you use dropfirst?]

pd.get_dummies(df.cholesterol, prefix='chol', prefix_sep='_')

pd.get_dummies(data=df.gluc, prefix='gluc', prefix_sep='_')

## Now that you know how to create these dummy variables, add them to the dataframe

df = pd.concat([df,
                 pd.get_dummies(df.cholesterol, prefix='chol', prefix_sep='_'),
                  pd.get_dummies(df.gluc, 'gluc', '_')],
                   axis=1)
df.head()

# Drop the original `cholesterol` and `gluc` columns

df.drop(['cholesterol', 'gluc'], axis = 1, inplace = True)
df.head()





# Split data into train and test sets

from sklearn.model_selection import train_test_split

# Predictor is `cardio` - presence or absence of heart disease

X = df.drop(['cardio'], axis = 1)
y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)




# Train the model

clf = DecisionTreeClassifier(max_depth=3, random_state=17)

heart_fit = clf.fit(X_train, y_train)


# Create Decisition Tree Graph

import graphviz

dot_data = export_graphviz(heart_fit, out_file=None, feature_names=X.columns, class_names=['Disease', 'Healthy'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph

#### You can also use your dot_file code here if the above doesn't work


# Make predictions

pred = heart_fit.predict(X_test)

## How accurate were our predictions?

accuracy_score(y_true=y_test, y_pred=pred)

#confusion_matrix(y_test, pred)

#classification_report(y_test, pred)



############## HYPER-PARAMETER TUNING ######################

# Perform cross-validated Grid Search using 5 folds

## See how accuracy changes with changing max_depth from 2 to 10
tree_params = {'max_depth': list(range(2, 11))}

accurate = make_scorer(accuracy_score)

tree_grid = GridSearchCV(clf, param_grid=tree_params, cv = 5)

grid = tree_grid.fit(X_train, y_train)





# Determine best estimator
tree_grid.best_estimator_
tree_grid.best_score_
tree_grid.best_params_

## Create model and assess accuracy of "best" model

accuracy_score(y_true=y_test, y_pred=tree_grid.predict(X_test))

tree_grid.cv_results_.keys()

params = tree_grid.cv_results_['params']
mean_acc = tree_grid.cv_results_['mean_test_score']

for mean, param in zip(mean_acc, params):
        print(f"{round(mean, 4)*100}% for {param}")

# Create plot of accuracy based on tree depth
sns.lineplot(x=np.arange(2, 11), y= mean_acc, label='Accuracy')
plt.title('Accuracy of Decision Tree by `max_depth`')
plt.show();






cross_val_score(tree_grid, X_train, y_train, cv=5)









############ SCORE paper ##################

df = pd.read_csv('data/mlbootcamp5_train.csv',
                 index_col='id', sep=';')

df.head()

# Convert `age` to be in years

df['age'] = round(df['age'] / 365)

# Binning - use `pd.cut`

# Bins for age: 1 is <50, 2 is 50-55, 3 is 55-60, 4 is 60+
df['age_bin'] = pd.cut(df['age'], [40, 50, 55, 60, 65])

# Bins for SBP
df['SBP_bin'] = pd.cut(df['ap_hi'], [120, 140, 160, 180])
df.head()

df['age_bin'].unique(), df['SBP_bin'].unique()

########### Convert Categorical to Numeric ############
#

############ Pandas Get Dummies ############

pd.get_dummies(df['age_bin'], prefix='age', prefix_sep='_')

############ Keras ###################

from keras.utils import to_categorical

encoded = to_categorical(df['cholesterol'])
encoded

argmax(encoded[0])

############ SKLearn #################

# Use `OneHotEncoder` if converting categorical data to numeric
# that does NOT have any hierarchical order
# This is like `pd.get_dummies`

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse=False, categories='auto')

#pd.get_dummies(df['cholesterol']) - is sooo much easier

# Reshape and Fit-Transform
## single brackets won't work, i.e. enc.fit_transform(df['cholesterol'].values)
## Alternative is: enc.fit_transform(df['cholesterol'].values.reshape(-1,1))
enc.fit_transform(df[['cholesterol']])

enc.get_feature_names()

#enc.fit_transform(df['cholesterol'].values.reshape(-1,1))

pd.DataFrame(enc.fit_transform(df[['cholesterol']].values),
             columns = [f"chol{i}" for i in enc.get_feature_names()])

# Create df, change the column names and then concatenate to add to the original df

enc.fit_transform(df[['age_bin']].dropna())

#
# LabelEncoder
# use if there is some order

le = LabelEncoder()

# Fit
le.fit(df['gender'])
le.classes_

# Transform
df['gender'] = le.transform(df['gender'])
df.head() # converted to 0 and 1 (from 1 and 2)

## Rename column to male
df.rename(columns = {'gender': 'male'}, inplace=True)


# Tip: Can perform the fit -> transform in 1 step using `fit_transform` method
