import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')

df.head()
df.info()


# Convert 'petal_length' to int (from float)
df['petal_length'] = df['petal_length'].astype(int)


# Convert `species` to categorical

df['species'] = df['species'].astype('category')