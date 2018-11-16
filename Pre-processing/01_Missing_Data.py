import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv')


# Drop all rows from a df containing missing values
df.dropna()


# Drop select rows
df.drop([1, 4, 5])



# Drop a column
df.drop("PassengerId", axis = 1)


# Drop columns where there are greater than 10 missing values
df.dropna(axis = 1, thresh = len(df)-10)


# Compute number of missing values in a column
df['Cabin'].isnull().sum()



# Return all rows where 'Cabin' has a value (i.e. non-null)
df[df['Cabin'].notnull()]










# Use missingno

msno.matrix(df); plt.show()