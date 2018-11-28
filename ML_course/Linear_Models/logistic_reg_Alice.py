import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns

# Import data

train_df = pd.read_csv('data/CatchMeIfYouCan_Alice/train_sessions.csv', index_col='session_id')

test_df = pd.read_csv('data/CatchMeIfYouCan_Alice/test_sessions.csv', index_col='session_id')

train_df.dtypes
test_df.dtypes

# Convert time columns to datetime dtypes

times = [f'time{i}' for i in range(1, 11)]

train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# Sort the data by time
train_df = train_df.sort_values(by='time1')

# Look at the first rows of the training set
train_df.head()


# Change site1, ..., site10 columns type to integer and fill NA-values with zeros
sites = [f'site{i}' for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype(np.uint16)
test_df[sites] = test_df[sites].fillna(0).astype(np.uint16)


train_df.shape
test_df.shape


# Load websites dictionary
with open(r"../coding_notes/data/CatchMeIfYouCan_Alice/site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)

# Create dataframe for the dictionary
sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])
print(u'Websites total:', sites_dict.shape[0])
sites_dict.head()



# EDA

# Top websites in the training data set
top_sites = pd.Series(train_df[sites].values.flatten()
                     ).value_counts().sort_values(ascending=False).head(5)
print(top_sites)
sites_dict.loc[top_sites.drop(0).index]


##### Time Featuring ######

# Create a separate dataframe where we will work with timestamps
time_df = pd.DataFrame(index=train_df.index)
time_df['target'] = train_df['target']

# Find sessions' starting and ending
time_df['min'] = train_df[times].min(axis=1)
time_df['max'] = train_df[times].max(axis=1)

# Calculate sessions' duration in seconds
time_df['seconds'] = (time_df['max'] - time_df['min']) / np.timedelta64(1, 's')

time_df.head()


time_df[time_df['target'] == 1] # Alice accounts for 2297 rows

len(time_df[time_df['target'] == 1]) / len(time_df) # Alice accounts for 0.9% of all sessions

time_df[time_df['target'] == 1]['seconds'].mean() # Her sessions last 52.3 seconds

time_df[time_df['target'] == 0]['seconds'].mean() # Others avg 139.3 secs

time_df[time_df['target'] == 1]['seconds'].describe()

time_df[time_df['target'] == 0]['seconds'].describe()

time_df[time_df['target'] == 1]['seconds'].var()

time_df[time_df['target'] == 0]['seconds'].var()

np.mean(time_df[time_df['target'] == 1]['seconds'] >= 40) # Only 24.1% of Alice sessions are >= 40 s




########## SITES -> SPARSE MATRIX ###########

#### Combine train and test df to do pre-processing on both simultaneously ###


# Our target variable
y_train = train_df['target']

# United dataframe of the initial data
full_df = pd.concat([train_df.drop('target', axis=1), test_df])

# Index to split the training and test data sets
idx_split = train_df.shape[0]


# Dataframe with indices of visited websites in session
full_sites = full_df[sites]
full_sites.head()




# "Flat" sequence of indices
sites_flatten = full_sites.values.flatten()

print(sites_flatten)

# Convert to a sparse data matrix
# (make sure you understand which of the `csr_matrix` constructors is used here)
# a further toy example will help you with it
full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0],
                                sites_flatten,
                                range(0, sites_flatten.shape[0]  + 10, 10)))[:, 1:]


full_sites_sparse.shape

# How much memory does a sparse matrix occupy?
print(f'{full_sites_sparse.count_nonzero()} elements * {8} bytes = {full_sites_sparse.count_nonzero() * 8} bytes')

# Or just like this:
print(f'Our sparse matrix size is {full_sites_sparse.data.nbytes} bytes')




# Train first model

def get_auc_lr_valid(X, y, C=1.0, seed=17, ratio = 0.9):
    # Split the data into the training and validation sets
    idx = int(round(X.shape[0] * ratio))
    # Classifier training
    lr = LogisticRegression(C=C, random_state=seed, solver='liblinear').fit(X[:idx, :], y[:idx])
    # Prediction for validation set
    y_pred = lr.predict_proba(X[idx:, :])[:, 1]
    # Calculate the quality
    score = roc_auc_score(y[idx:], y_pred)

    return score


# Select the training set from the united dataframe (where we have the answers)
X_train = full_sites_sparse[:idx_split, :]

# Calculate metric on the validation set
print(get_auc_lr_valid(X_train, y_train))





# Function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)




# Train the model on the whole training data set
# Use random_state=17 for repeatability
# Parameter C=1 by default, but here we set it explicitly
lr = LogisticRegression(C=1.0, random_state=17, solver='liblinear').fit(X_train, y_train)

# Make a prediction for test data set
X_test = full_sites_sparse[idx_split:,:]
y_test = lr.predict_proba(X_test)[:, 1]

# Write it to the file which could be submitted
write_to_submission_file(y_test, 'baseline_1.csv') # Got an AUC ROC of 0.90812



# What years are present in the dataset?
train_df.columns

train_df['time1'].apply(lambda row: row.year).min() #2013
train_df['time1'].apply(lambda row: row.year).max() # 2014




###### TIME FEATURING CONT'D ############

# Dataframe for new features
full_new_feat = pd.DataFrame(index=full_df.index)

# Add start_month feature
full_new_feat['start_month'] = full_df['time1'].apply(lambda ts:
                                                      100 * ts.year + ts.month).astype('float64')


full_new_feat['start_month'].unique()

graphs = pd.concat([full_new_feat[:idx_split], time_df['target']], axis=1)

#sns.scatterplot(x='start_month', y='target', hue='target', data=graphs); plt.show()

sns.countplot(x='start_month', hue='target', data=graphs); plt.show()



graphs.loc[graphs['target'] == 0, 'start_month'].hist(label = "Others")
graphs.loc[graphs['target'] == 1, 'start_month'].hist(label = "Alice")
plt.xlabel('Year/Month')
plt.legend()
plt.show();


# Alice only

plt.hist(x='start_month', data=graphs[graphs['target'] == 1], label='Alice');
plt.legend()
plt.show();




# Train with `start_month` variable

# Add the new feature to the sparse matrix
tmp = full_new_feat[['start_month']].values
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:]]))

# Compute the metric on the validation set
print(get_auc_lr_valid(X_train, y_train)) # AUC is 0.75





### Why did our model perform poorly after adding a useful feature such as 'start_month'?
# Answer: Scaling!


# Add the new standardized feature to the sparse matrix
tmp = StandardScaler().fit_transform(full_new_feat[['start_month']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:]]))

# Compute metric on the validation set
print(get_auc_lr_valid(X_train, y_train)) # AUC improved to 0.9197




# Add in a Number of Unique sites feature
## Gives the number of unique sites visited during an individual session

full_df[sites].head() # expect 2, 2, 6

# full_df[sites].head().apply(lambda x: x.nunique(), axis = 1) ## this treats 0 as unique

# I converted 0 to NaN, then performed nunique for each row

full_df[sites].head().replace(0, np.nan).apply(lambda x: x.nunique(), axis = 1)

full_new_feat['n_unique_sites'] = full_df[sites].replace(0, np.nan).apply(lambda x: x.nunique(), axis = 1)

full_new_feat.head()





## How does the model perform now?

# Scale the 'start_month' and Add the 'n_unique_sites' feature
tmp1 = StandardScaler().fit_transform(full_new_feat[['start_month']])
tmp2 = full_new_feat[['n_unique_sites']].values
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp1[:idx_split,:], tmp2[:idx_split,:]]))

# Compute metric on the validation set
print(get_auc_lr_valid(X_train, y_train)) # AUC decreased to 0.9168






# 2 new features: 'start_hour' and 'morning'

full_new_feat['start_hour'] = full_df['time1'].apply(lambda row: row.hour)

#(full_new_feat['start_hour'].head() <= 11).astype(int)
full_new_feat['morning'] = (full_new_feat['start_hour'] <= 11).astype(int)

full_new_feat.head()



## How does the model perform now?

# Scale the 'start_month' and Add the 'start_hour' and 'morning' features
tmp1 = StandardScaler().fit_transform(full_new_feat[['start_month']])
tmp2 = full_new_feat[['start_hour', 'morning']].values
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp1[:idx_split,:], tmp2[:idx_split,:]]))

# Compute metric on the validation set
print(get_auc_lr_valid(X_train, y_train)) # AUC improved to 0.9585

## 'start_hour' alone improves to 0.9573
## 'morning' alone improves to 0.9487






############ HYPERPARAMETER TUNING ##################

## Default: C = 1

# Compose the training set
tmp_scaled = StandardScaler().fit_transform(full_new_feat[['start_month',
                                                           'start_hour',
                                                           'morning']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:],
                             tmp_scaled[:idx_split,:]]))

# Capture the quality with default parameters
score_C_1 = get_auc_lr_valid(X_train, y_train)
print(score_C_1)



## Adjust C, i.e. the regularization parameter

from tqdm import tqdm

# List of possible C-values
Cs = np.logspace(-3, 1, 10)
scores = []
for C in tqdm(Cs):
    scores.append(get_auc_lr_valid(X_train, y_train, C=C))

scores

# Which C gives the max score?

C_dict = {} # a dict to hold C: score
for C, score in zip(Cs, scores):
    C_dict[C] = score

for k,v in C_dict.items():
    print(f"{round(k, 4)}: {round(v*100, 3)}%")

max(C_dict, key=C_dict.get) # gives the max C
C_dict[max(C_dict, key=C_dict.get)] # gives the associated value w/ max C

C = max(C_dict, key=C_dict.get)


# Plot AUC vs C (regularization parameter)

plt.plot(Cs, scores, 'ro-')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('AUC-ROC')
plt.title('Regularization Parameter Tuning')
# horizontal line -- model quality with default C value
plt.axhline(y=score_C_1, linewidth=.5, color='b', linestyle='dashed')
plt.show();






# Use the optimal parameter to generate predictions for test set

# Prepare the training and test data
tmp_scaled = StandardScaler().fit_transform(full_new_feat[['start_month', 'start_hour',
                                                           'morning']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:],
                             tmp_scaled[:idx_split,:]]))
X_test = csr_matrix(hstack([full_sites_sparse[idx_split:,:],
                            tmp_scaled[idx_split:,:]]))

# Train the model on the whole training data set using optimal regularization parameter
lr = LogisticRegression(C=C, random_state=17, solver='liblinear').fit(X_train, y_train)

# Make a prediction for the test set
y_test = lr.predict_proba(X_test)[:, 1]


# Write it to the submission file

# Function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)

write_to_submission_file(y_test, 'baseline_2.csv')








# TimeSeriesSplit for Time Series Cross Validation

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain

# Re-initialize and setup train and test sets
times = [f'time{i}' for i in range(1, 11)]

train_df = pd.read_csv('data/CatchMeIfYouCan_Alice/train_sessions.csv', index_col='session_id', parse_dates=times)
test_df = pd.read_csv('data/CatchMeIfYouCan_Alice/test_sessions.csv', index_col='session_id', parse_dates=times)

train_df = train_df.sort_values(by='time1')

sites = [f'site{i}' for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype(np.uint16)
test_df[sites] = test_df[sites].fillna(0).astype(np.uint16)

# Need to get things into a format that CountVectorizer will take
# I tried to flatten and a bunch of different string formatting and lists of lists stuff, but to no avail
train_df[sites].to_csv('train_sessions_text.txt', sep=' ', index=None, header=None)
test_df[sites].fillna(0).astype('int').to_csv('test_sessions_text.txt', sep=' ', index=None, header=None)

cv = CountVectorizer(ngram_range=(1, 3), max_features=50000)

with open('train_sessions_text.txt') as inp_train_file:
    X_train = cv.fit_transform(inp_train_file)
with open('test_sessions_text.txt') as inp_test_file:
    X_test = cv.transform(inp_test_file)
X_train.shape, X_test.shape

y_train = train_df['target'].astype('int')

tscv = TimeSeriesSplit(n_splits=10)


[(el[0].shape, el[1].shape) for el in tscv.split(X_train)]

logit = LogisticRegression(C=1, random_state=17)

import warnings
warnings.filterwarnings('ignore')

cv_scores = cross_val_score(logit, X_train, y_train, cv=tscv, scoring='roc_auc', n_jobs=1)
cv_scores, cv_scores.mean()

logit.fit(X_train, y_train)

logit_test_pred = logit.predict_proba(X_test)[:, 1]


### Add Time Features

def add_time_features(df, X_sparse):
    hour = df['time1'].apply(lambda ts: ts.hour)
    morning = ((hour >= 7) & (hour <= 11)).astype('int')
    day = ((hour >= 12) & (hour <= 18)).astype('int')
    evening = ((hour >= 19) & (hour <= 23)).astype('int')
    night = ((hour >= 0) & (hour <= 6)).astype('int')
    X = hstack([X_sparse, morning.values.reshape(-1, 1),
                day.values.reshape(-1, 1), evening.values.reshape(-1, 1),
                night.values.reshape(-1, 1)])
    return X

X_train_new = add_time_features(train_df.fillna(0), X_train)
X_test_new = add_time_features(test_df.fillna(0), X_test)

X_train_new.shape, X_test_new.shape


cv_scores = cross_val_score(logit, X_train_new, y_train, cv=time_split, scoring='roc_auc', n_jobs=1)
cv_scores, cv_scores.mean() # see an improvement w/ the new time features

logit.fit(X_train_new, y_train)
logit_test_pred2 = logit.predict_proba(X_test_new)[:, 1]




# Create regularization penalty space

penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)


# Create grid search using Time Series Split CV

log_grid = GridSearchCV(logit, hyperparameters, cv=tscv, scoring='roc_auc', verbose=True, n_jobs=-1)
log_grid.fit(X_train_new, y_train)

log_grid.best_score_
log_grid.best_params_


logit_test_pred3 = log_grid.predict_proba(X_test_new)[:, 1]
write_to_submission_file(logit_test_pred3, 'subm3.csv')
