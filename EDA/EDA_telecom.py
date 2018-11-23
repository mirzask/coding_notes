# Import libraries

import pandas as pd
import numpy as np

# Read data

df = pd.read_csv('data/telecom_churn.csv')

# Dimensions/Shape of data
df.shape

# List column names
df.columns

# View columns and data types for each column
df.info()

# Change data type using `astype` method
df['Churn'] = df['Churn'].astype('int64')

# Descriptive stats with `describe()`
df.describe()

# See stats for non-numerical data using `include=`
# For categorical use `object`
# For boolean use `bool`

df.describe(include=['object', 'bool'])

# Counts for each values
df['Churn'].value_counts()

# Values as a proportion of the whole
df['Churn'].value_counts(normalize=True)

# Create a NEW column

total_calls = df['Total day calls'] + df['Total eve calls'] + df['Total night calls'] + df['Total intl calls']
df.insert(loc=len(df.columns), column='Total calls', value=total_calls)
df.head()

# Alternative
df['Total calls'] = total_calls
df.head()

# Remove a column with `drop`
df.drop(labels='Total calls', axis=1, inplace=True)
df.head()

# Sort by variable using `sort_values`
df.sort_values(by='Total day charge', ascending=False).head()

df.sort_values(by=['Churn', 'Total day charge'], ascending=[True, False]).head()

# Indexing

# Select the first 5 rows and columns 'State' to 'Area Code'
df.loc[0:4, 'State':'Area code']
df.iloc[0:5, 0:3]

# Select based on starts with certain character
## Example, select states that begin with the letter 'W'

df[df['State'].str.startswith('W')].head()


df[df['State'].apply(lambda state: state[0] == 'W')].head()


# Replace values
## Convert 'Yes' and 'No' to True and False
d = {'Yes': True, 'No': False}
df['International plan'].head()
df['International plan'] = df['International plan'].map(d)
df['International plan'].head()

# Alternative approach: use `replace`
df['Voice mail plan'].head()
df.replace({'Voice mail plan': d}, inplace=True)
df.head()

# Groupby
## Calculate the mean, std, max and min for each group
df.groupby(by='Churn')['Total day minutes', 'Total eve minutes', 'Total night minutes'].agg([np.mean, np.std, np.max, np.min])

# Cross tabulation between variables

pd.crosstab(df['Churn'], df['International plan'])

pd.crosstab(df['Churn'], df['International plan'], normalize=True)

pd.crosstab(df['Churn'], df['Customer service calls'], margins=True)

# Pivot Table

# Minute use by time of day vs Area Code

pd.pivot_table(data=df, values=['Total day calls', 'Total eve calls', 'Total night calls'], index='Area code', aggfunc='mean')
