# Random Forest notes

Below are the parameters which we need to pay attention to when we are building a new model:

- `n_estimators` — the number of trees in the forest;
- `criterion` — the function used to measure the quality of a split;
- `max_features` — the number of features to consider when looking for the best split;
- `min_samples_leaf` — the minimum number of samples required to be at a leaf node;
- `max_depth` — the maximum depth of the tree.

# Rules of thumb

While building decision trees using Random Forest, for each split, we first randomly pick $m$ features from the $d$ original ones and then search for the next best split only among the subset.

For classification problems, it is advisable to set $m = \sqrt{d}$. For regression problems, we usually take $m = \frac{d}{3}$, where $d$ is the number of features. 

It is recommended to build each tree until all of its leaves contain only $n_{min}=1$ examples for classification and $n_{min}=5$ examples for regression.