import seaborn as sns
import matplotlib.pyplot as plt

wine = pd.read_csv('https://raw.githubusercontent.com/hadley/rminds/master/1-data/wine.csv').drop('type', axis = 1)

# Create correlation matrix

sns.heatmap(wine.corr(), cmap='viridis'); plt.show()

plt.matshow(wine.corr(), cmap = 'viridis')
plt.colorbar()
plt.show()