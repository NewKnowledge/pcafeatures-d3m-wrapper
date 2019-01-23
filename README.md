# Pcafeatures D3M Wrapper
Wrapper of the Punk feature ranker into D3M infrastructure. All code is written in Python 3.5 and must be run in 3.5 or greater.

The base library can be found here: https://github.com/NewKnowledge/punk.

## Output
The output is a DataFrame with two columns. The first column is an ordered list of all original features, ordered by their contribution to the first principal component. The second column is the feature's contribution to the first principal component, which serves as a proxy for the 'score' of that feature. 

## Available Functions

#### produce
Perform principal component analysis on all numeric data in the dataset and then use each original features contribution to the first principal component as a proxy for the 'score' of that feature. Returns a dataframe containing an ordered list of all original features as well as their contribution to the first principal component.
