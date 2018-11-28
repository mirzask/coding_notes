import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

# Load data

data = pd.read_csv('data/credit_scoring_sample.csv', sep=";")

data.head()



# Function to impute NaN values w/ median for each column

def impute_nan_with_median(table):
    for col in table.columns:
        table[col]= table[col].fillna(table[col].median())
    return table


data.dtypes

# How is our missing data problem?

data.isnull().sum()

msno.matrix(data)
msno.bar(data)


# What is the distribution of our target variable, SeriousDlqin2yrs

ax = data['SeriousDlqin2yrs'].hist(orientation='horizontal', color='red')
ax.set_xlabel("number_of_observations")
ax.set_ylabel("unique_value")
ax.set_title("Target distribution")

print('Distribution of target:')
data['SeriousDlqin2yrs'].value_counts() / data.shape[0]

data['SeriousDlqin2yrs'].value_counts(normalize=True)



# Impute median for missing values

table = impute_nan_with_median(data)

X = table.drop('SeriousDlqin2yrs', axis=1)
y = table['SeriousDlqin2yrs']




###### BOOTSTRAP ########

# Generate 90% CI of monthly income for those w/ overdue loans vs those that pay on time

import numpy as np

def get_bootstrap_samples(data, n_samples):
    """Generate bootstrap samples using the bootstrap method."""
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples

def stat_intervals(stat, alpha):
    """Produce an interval estimate."""
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries

table['SeriousDlqin2yrs'].nunique()

bad_income = table[table['SeriousDlqin2yrs'] == 1]['MonthlyIncome'].values

good_income = table[table['SeriousDlqin2yrs'] == 0]['MonthlyIncome'].values

# Set the seed for reproducibility of the results
np.random.seed(17)

# Generate the samples using bootstrapping and calculate the mean for each of them
bad_income_means = [np.mean(sample)
                       for sample in get_bootstrap_samples(bad_income, 1000)]
good_income_means = [np.mean(sample)
                       for sample in get_bootstrap_samples(good_income, 1000)]
# Print the resulting interval estimates
print("Monthly income from defaulters: mean interval",
      stat_intervals(bad_income_means, 0.10))
print("Monthly income from on-time payers: mean interval",
stat_intervals(good_income_means, 0.10))

# Find the difference good_income_lower−bad_income_upper
round(stat_intervals(good_income_means, 0.10)[0] - stat_intervals(bad_income_means, 0.10)[1], 0)




############# HYPERPARAMETER TUNING ############

#### DECISION TREE ####


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


dt = DecisionTreeClassifier(random_state=17, class_weight='balanced')

max_depth_values = [5, 6, 7, 8, 9]
max_features_values = [4, 5, 6, 7]
tree_params = {'max_depth': max_depth_values,
               'max_features': max_features_values}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)


gcv = GridSearchCV(dt, tree_params, n_jobs=-1, cv=skf, scoring='roc_auc' , verbose=1)
gcv.fit(X, y)

gcv.best_score_
gcv.best_params_


# Was Cross-validation stable under optimal combinations of hyperparameters?
## We call cross-validation stable if the standard deviation of the metric
## on the cross-validation is less than 1%

gcv.cv_results_['std_train_score'] > 0.01






############# Random Forest ################

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rfc = RandomForestClassifier(n_estimators=100, random_state=17,
                             class_weight='balanced', max_depth=7,
                             max_features=6, n_jobs=-1, oob_score=True)

cross_val_score(rfc, X, y, scoring='roc_auc', cv=5, verbose=1)

np.mean([0.82641196, 0.84091797, 0.82461536, 0.83676623, 0.83008748])



############# HYPERPARAMETER TUNING ###############

max_depth_values = range(5, 15)
max_features_values = [4, 5, 6, 7]
forest_params = {'max_depth': max_depth_values,
                'max_features': max_features_values}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

rfc = RandomForestClassifier(n_estimators=100, random_state=17,
                             class_weight='balanced', n_jobs=-1)

gcv = GridSearchCV(rfc, forest_params, n_jobs=-1, cv=skf, scoring='roc_auc' , verbose=1)
gcv.fit(X, y)


gcv.best_params_
