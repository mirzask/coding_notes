# Choose the numeric features
cols = []

for i in df.columns:
    if (df[i].dtype == "float64") or (df[i].dtype == 'int64'):
        cols.append(i)


# Divide the dataset into the input and target
X, y = df[cols].copy(), np.asarray(df["target_col"], dtype='int8')