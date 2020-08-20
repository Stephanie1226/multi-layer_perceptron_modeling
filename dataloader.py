from __future__ import division  # floating point division
import numpy as np

def load_data(trainsize=15000, testsize=5000):
    """ A physics classification dataset """
    filename = 'datasets/dataset.csv'
    dataset = loadcsv(filename)
    trainset, testset = splitdataset(dataset, trainsize, testsize)
    return trainset, testset

def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset

def splitdataset(dataset, trainsize, testsize, testdataset=None, featureoffset=None, outputfirst=None):
    # Generate random indices without replacement, to make train and test sets disjoint
    randindices = np.random.choice(dataset.shape[0], trainsize + testsize, replace=False)
    featureend = dataset.shape[1] - 1
    outputlocation = featureend
    if featureoffset is None:
        featureoffset = 0
    if outputfirst is not None:
        featureoffset = featureoffset + 1
        featureend = featureend + 1
        outputlocation = 0

    Xtrain = dataset[randindices[0:trainsize], featureoffset:featureend]
    ytrain = dataset[randindices[0:trainsize], outputlocation]
    Xtest = dataset[randindices[trainsize:trainsize + testsize], featureoffset:featureend]
    ytest = dataset[randindices[trainsize:trainsize + testsize], outputlocation]

    if testdataset is not None:
        Xtest = dataset[:, featureoffset:featureend]
        ytest = dataset[:, outputlocation]

    # Normalize features, with maximum value in training set
    for ii in range(Xtrain.shape[1]):
        maxval = np.max(np.abs(Xtrain[:, ii]))
        if maxval > 0:
            Xtrain[:, ii] = np.divide(Xtrain[:, ii], maxval)
            Xtest[:, ii] = np.divide(Xtest[:, ii], maxval)

    return ((Xtrain, ytrain), (Xtest, ytest))

