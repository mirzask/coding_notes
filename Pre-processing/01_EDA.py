import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')

# List dataset feature names
print(df.columns)

# List dataset datatypes
print(df.dtypes)


# Generate basic stats
df.describe()


# Use DataFrameSummary (from pandas_summary)
#from pandas_summary import DataFrameSummary

#DataFrameSummary(df).summary()
