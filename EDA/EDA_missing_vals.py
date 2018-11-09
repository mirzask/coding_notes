import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt


titanic = pd.read_csv('https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv')



# Matrix

msno.matrix(titanic); plt.show()


# Bar chart

msno.bar(titanic); plt.show()

msno.bar(titanic, log=True); plt.show() # logarithmic scale


# Heatmap - any link between NA between features?

msno.heatmap(titanic); plt.show()


# Dendrogram

msno.dendrogram(titanic); plt.show()


