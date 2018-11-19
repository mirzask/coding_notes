import pandas as pd

address = pd.read_csv("https://raw.githubusercontent.com/codeforamerica/ohana-api/master/data/sample-csv/addresses.csv")

# Extracting numbers from strings
# using RegEx

import re

# Write a pattern to extract numbers and decimals
def return_address_number(address):
    pattern = re.compile(r"\d+")
    
    # Search the text for matches
    add = re.match(pattern, address)
    
    # If a value is returned, use group(0) to return the found value
    if add is not None:
        return int(add.group(0))
        
# Apply the function to the Length column and take a look at both columns
address["add_number"] = address['address_1'].apply(lambda row: return_address_number(row))
print(address[["address_1", "add_number"]].head(10))


############# TITANIC TILE EXTRACTION ######################


df = pd.read_csv('https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv')

df.columns

# Extract title from passenger names

# Write a pattern to extract numbers and decimals
def title_extractor(full_name):
    pattern = re.compile(r"(\w+)\.")
    
    # Search the text for matches
    add = re.search(pattern, full_name)
    
    # If a value is returned, use group(0) to return the found value
    if add is not None:
        return add.group(1)

df["title"] = df['Name'].apply(lambda row: title_extractor(row))
print(df[["Name", "title"]].head(10))

df.title.value_counts()





# Vectorizing text
## tf/idf - method for weighting of each word
### tf = term frequency (# of times word appears in doc)
### idf = inverse document frequency (# of docs / total # of docs containing term t)
# Source: https://www.kaggle.com/edchen/tf-idf

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

corpus = [
 'the brown fox jumped over the brown dog',
 'the quick brown fox',
 'the brown brown dog',
 'the fox ate the dog'
]

X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names())
print(X.toarray())


print(X.shape)
print(tfidf.vocabulary_) # dict w/ vocabulary and index value of each

vocab = {v:k for k, v in tfidf.vocabulary_.items()}








