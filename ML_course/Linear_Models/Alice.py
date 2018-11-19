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

test_df = pd.read_csv('../../data/test_sessions.csv',
                      index_col='session_id')

train_df = pd.read_csv('../coding_notes/data/CatchMeIfYouCan_Alice/train_sessions.csv', index_col='session_id')

test_df = pd.read_csv('../coding_notes/data/CatchMeIfYouCan_Alice/test_sessions.csv', index_col='session_id')

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


full_new_feat['start_month']

graphs = pd.concat([full_new_feat[:idx_split], time_df['target']], axis=1)

sns.scatterplot(x='start_month', y='target', hue='target', data=graphs); plt.show()

sns.countplot(x='start_month', hue='target', data=graphs); plt.show()

sns.countplot(x='start_month', hue = 'target', data=graphs[graphs.target == 1]); plt.show()