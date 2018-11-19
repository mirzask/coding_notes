import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')

df.info()

[f'run{i}' for i in range(1, 6)]

run_columns = ['run%s' % i for i in range(1, 6)]
print(run_columns)



[f'site{i}' for i in range(1, 11)]

sites = ['site%s' % i for i in range(1, 11)]
print(sites)



# LABEL ENCODER

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['species_encoded'] = le.fit_transform(df['species']) # convert to numeric labels

le.classes_
le.inverse_transform(df['species_encoded']) # converts from numerics back





# ONE HOT ENCODING

#### Using pd.get_dummies()

species_1hot = pd.get_dummies(df['species'])
species_1hot.head() # returns a pandas df

### Using sklearn

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

species_ohe = ohe.fit_transform(df[['species']])
species_ohe # returns an n-d array

ohe.get_feature_names() # gives the categorical names


# NOTE: ohe.fit_transform(df['species']) does NOT work. sklearn needs to be 2D
# df['species'].ndim ---> 1
# df[['species']].ndim ---> 2