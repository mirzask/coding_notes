############# SUMMARY STATS ################

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

iris = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')

sns.set(style='darkgrid')

# Mean

mean_petal = np.mean(iris['petal_length'])
mean_petal

# Median - the 50th percentile

median_petal = np.median(iris['petal_length'])
median_petal


# Print values using f-string
f'The mean petal length is {np.round(mean_petal, 3)}, whereas the median petal length is {median_petal}.'



# Variance

var_petal = np.var(iris['petal_length'])


# Standard deviation

sd_petal = np.std(iris['petal_length'])


# Print var and std using f-string
f'The variance of petal length is {np.round(var_petal, 2)} and the standard deviation is {np.round(sd_petal, 2)}.'



# Covariance

## Using numpy 

versicolor = iris[iris['species'] == 'versicolor']

cov_matrix = np.cov(versicolor.petal_length, versicolor.petal_width)
cov_matrix

f'The covariance between versicolor petal length and width is {np.round(cov_matrix[0,1], 3)}.'

## Using Pandas

cov_matrix = versicolor[['petal_length', 'petal_width']].cov()
cov_matrix


## Scatterplot

sns.scatterplot(x='petal_length', y='petal_width', data=versicolor); plt.show()

# Pearson correlation coefficient - unitless

# Pearson (rho) = Covariance / ((sd of x)*(sd of y))
# i.e. variability d/t codependence  /  indep variability
## Interpretation: no correltion -> rho = 0, perfect correlation -> abs(rho) = 1

from scipy.stats import pearsonr

rho = pearsonr(versicolor.petal_length, versicolor.petal_width)
rho

f'The Pearson coefficient for versicolor petal length and width is {np.round(rho[0], 3)}, with a p-value of {rho[1]}.'






# Percentiles

## Calculate the 25th, 50th and 75th percentiles

percentiles = np.percentile(iris['petal_length'], [2.5, 25, 50, 75, 97.5]) 
percentiles

## Boxplot

sns.boxplot(iris['species'], iris['petal_length']); plt.show()

sns.boxplot('species', 'petal_length', data=iris); plt.show()




## ECDF


####### Function

def ecdf(data):
    """
    Compute ECDF for a one-dimensional array of measurements. `n` is the number of data points
    - `x` is the x-axis data for the ECDF
    - `y` is the y-axis data for the ECDF
    """
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n + 1) / n
    return x, y


#### ECDF plot

x, y = ecdf(iris['petal_length'])
_ = plt.plot(x, y, marker = '.', linestyle = 'none')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Overlay the percentiles
_ = plt.plot(percentiles, np.array([2.5, 25, 50, 75, 97.5]) / 100, marker = 'D', color = 'red', linestyle = 'none')

plt.margins(0.02)
plt.show()