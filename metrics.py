## DataFrames should have "Tags" and "New_Tags" column.
## "Tags" are original tags
## "New_Tags" will be predicted tags

## USAGE
## DF.apply(accuracyFunc, axis=1)
## DF.apply(recallFunc, axis=1)
## DF.apply(precisionFunc, axis=1)

## ----- OR ----- (You can make a new column to store these metrics in new column)
## DF['accuracy'] = new_DF.apply(accuracyFunc, axis=1)
## DF['recallRate'] = new_DF.apply(recallFunc, axis=1)
## DF['precisionRate'] = new_DF.apply(precisionFunc, axis=1)

import pandas as pd


def accuracyFunc(row):
    return 1 if any(i in row['New_Tags'].split(',') for i in row['Tags'].split(',')) else 0


def recallFunc(row):
    actualTags = row.Tags.split(",")
    predictedTags = row.New_Tags.split(",")
    
    # Tags predicted that are correct.
    recalledTags = set(actualTags).intersection(predictedTags)
    # Number of correctly predicted Tags
    numberOfRecalledTags = len(recalledTags)
    
    return numberOfRecalledTags/len(actualTags) if numberOfRecalledTags else 0


def precisionFunc(row):
    actualTags = row.Tags.split(",")
    predictedTags = row.New_Tags.split(",")
    
    # Tags predicted that are correct.
    recalledTags = set(actualTags).intersection(predictedTags)
    # Number of correctly predicted Tags
    numberOfRecalledTags = len(recalledTags)
    numberOfPredictedTags = len(predictedTags)
    
    return numberOfRecalledTags/numberOfPredictedTags if numberOfRecalledTags else 0