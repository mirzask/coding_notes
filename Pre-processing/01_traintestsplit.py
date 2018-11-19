import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv')

df.columns

# Create X and y

############# Method 1 #################


X = df.drop('Survived', axis=1)

y = df['Survived']


########### Method 2 ###################

y = df.pop('Survived').values


df.columns # notice how 'Survived' column is missing

X = df

##########################################






y.value_counts(normalize = True)

# Classic method
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

y_train.value_counts(normalize = True)


# Stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify = y)

y_train.value_counts(normalize = True)