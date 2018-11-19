import pandas as pd

# PCA

wine = pd.read_csv('https://raw.githubusercontent.com/hadley/rminds/master/1-data/wine.csv').drop('type', axis = 1)

from sklearn.decomposition import PCA


pca = PCA()
wine_pca = pca.fit_transform(wine)

#print(wine_pca)
print(pca.explained_variance_ratio_) # we can maybe drop components that don't explain much variance

