import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

iris = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')




# Plots

# Matplotlib 
# 
# Histogram

_ = plt.hist(iris[iris.species == 'versicolor']['petal_length'])
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('count')
plt.show()


## ECDF

#### Function ####

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

#####################

#### ECDF plot

import numpy as np
x, y = ecdf(iris['petal_length'])
_ = plt.plot(x, y, marker = '.', linestyle = 'none')
plt.margins(0.02)
plt.show()



### Multiple ECDFs - useful for comparison

# Compute ECDFs
x_set, y_set = ecdf(iris[iris.species == 'setosa']['petal_length'])
x_vers, y_vers = ecdf(iris[iris.species == 'versicolor']['petal_length'])
x_virg, y_virg = ecdf(iris[iris.species == 'virginica']['petal_length'])



# Plot all ECDFs on the same plot
_ = plt.plot(x_set, y_set, marker = '.', linestyle = 'none')
_ = plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none')
_ = plt.plot(x_virg, y_virg, marker = '.', linestyle = 'none')

# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()






# Seaborn 

## Set global settings w/ `sns.set()`
sns.set(style="whitegrid")


## Plot everything

sns.pairplot(iris, hue='species'); plt.show()


# ECDF

sns.kdeplot(iris['petal_length'], cumulative=True); plt.show()

# ECDF + histogram superimposed
sns.distplot(iris['petal_length'], hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True)); plt.show()



## Histogram

sns.distplot(iris[iris.species == 'versicolor']['petal_length']); plt.xlabel('petal length (cm)'); plt.ylabel('count'); plt.show()



## Bee-Swarm plot

_ = sns.swarmplot(x='species', y='sepal_length', data=iris)
_ = plt.xlabel('Species')
_ = plt.ylabel('Petal length (cm)')
plt.show()



