import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas_profiling

iris = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')

iris.head()
iris.shape
iris.columns
iris.dtypes
iris.info()
iris.describe()
iris.describe(include = ['object', 'bool'])

iris['species'].value_counts()
iris['species'].value_counts(normalize = True)

# Change column data type with `astype`
iris['sepal_length'].astype('int64')
iris['species'] = iris['species'].astype('category')


# Apply functions to elements
iris.apply(np.max)




# Group by
iris.groupby(by = ['species'])['sepal_length'].mean()

## Use `agg` for list of functions
iris.groupby(by = ['species'])['sepal_length'].agg([np.mean, np.max, np.min])


# Tables

pd.crosstab(iris['species'], iris['sepal_length'] < 5.2)






# Generate Pandas Profiling HTML

profile = pandas_profiling.ProfileReport(iris)
profile.to_file(outputfile="myoutputfile.html")