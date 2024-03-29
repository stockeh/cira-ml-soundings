import matplotlib.pyplot as plt
import matplotlib.patches as pltpatch  # for Arc
import matplotlib.collections as pltcoll

import pandas as pd
import numpy as np

######################################################################
# Machine Learning Utilities.
#
#  percent_correct
#  rmse
#  partition
#  confusion_matrix
#  draw
######################################################################


def percent_correct(actual, predicted):
    return 100 * np.mean(actual == predicted)

######################################################################


def rmse(A, B):
    return np.sqrt(np.mean((A - B)**2))

######################################################################


def standard_partition_indicies(files, percentages=(0.75,0.10,0.15),
                                shuffle=False, seed=1234):
    """Partition data to have an equal proportion of locations per data set.
    ---
    params:
        percentages : list
            (train, validation, test)
    """
    validateFraction = 0
    if isinstance(percentages, (tuple, list)):
        trainFraction = percentages[0]
        testFraction = percentages[-1]
        if len(percentages) == 3:
            validateFraction = percentages[1]

    elif isinstance(percentages, float):
        trainFraction = percentages
        testFraction = 1 - trainFraction

    else:
        raise TypeError(
            f'percentages {percentages} must be of the following (train, val, test) or 0.8 for train')
    
    if shuffle:
        np.random.seed(seed)

    content = np.array([f.split('_') for f in files])
    locs    = content[:,0:1]
    mons    = pd.to_datetime(content[:,1], format='%Y-%m-%dT%H:%M:%S.%f').month.values.reshape(-1,1)
    classes = np.unique(locs)
    months  = np.unique(mons)

    train_i = []
    val_i   = []
    test_i  = []

    for c in classes:
        # all indicies for class c
        c_i = np.where(locs == c)[0]
        for m in months:
            # all indicies for month m
            m_i  = np.where(mons == m)[0]
            # intersection between month and location indicies
            cm_i = np.intersect1d(c_i, m_i)
            
            if shuffle: # shuffle indicies
                np.random.shuffle(cm_i)
            
            # partitioned indicies for class c
            n = len(cm_i)
            nTrain    = round(trainFraction * n)
            nValidate = round(validateFraction * n)
            nTest     = round(testFraction * n)
            if nTrain + nValidate + nTest > n:
                nTest = n - nTrain - nValidate

            train_i += cm_i[:nTrain].tolist()
            if nValidate > 0:
                val_i += cm_i[nTrain:nTrain+nValidate].tolist()
            test_i += cm_i[nTrain+nValidate:nTrain+nValidate+nTest].tolist()

    if shuffle: # shuffle all indicies
        np.random.shuffle(train_i)
        np.random.shuffle(val_i)
        np.random.shuffle(test_i)

    if validateFraction > 0:
        return train_i, val_i, test_i
    else:
        return train_i, test_i
    

def parition_all(rap=None, raob=None, goes=None, rtma=None, 
                 percentages=(0.75,0.10,0.15), shuffle=False, seed=1234):
    """
    ---
    params:
        percentages : list
            (train, validation, test)
    """
    
    print('DEPRECATED! Use ml.standard_partition_indicies')
    
    validateFraction = 0
    if isinstance(percentages, (tuple, list)):
        trainFraction = percentages[0]
        testFraction = percentages[-1]
        if len(percentages) == 3:
            validateFraction = percentages[1]

    elif isinstance(percentages, float):
        trainFraction = percentages
        testFraction = 1 - trainFraction

    else:
        raise TypeError(
            f'percentages {percentages} must be of the following (train, val, test) or 0.8 for train')
    
    n = raob.shape[0]
    nTrain = round(trainFraction * n)
    nValidate = round(validateFraction * n)
    nTest = round(testFraction * n)
    if nTrain + nValidate + nTest > n:
        nTest = n - nTrain - nValidate
        
    rowIndices = np.arange(n)
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(rowIndices)
        
    RAPtrain  = None
    RAPval    = None
    RAPtest   = None
        
    RTMAtrain = None
    RTMAval   = None
    RTMAtest  = None

    GOEStrain = None
    GOESval   = None
    GOEStest  = None
    
    RAOBtrain = None
    RAOBval   = None
    RAOBtest  = None
    
    if rap is not None:
        RAPtrain = rap[rowIndices[:nTrain], :]
        if nValidate > 0:
            RAPval = rap[rowIndices[nTrain:nTrain+nValidate], :]
        RAPtest = rap[rowIndices[nTrain+nValidate:nTrain+nValidate+nTest], :]
           
    if rtma is not None:
        RTMAtrain = rtma[rowIndices[:nTrain], :]
        if nValidate > 0:
            RTMAval = rtma[rowIndices[nTrain:nTrain+nValidate], :]
        RTMAtest = rtma[rowIndices[nTrain+nValidate:nTrain+nValidate+nTest], :]
        
    if goes is not None:
        GOEStrain = goes[rowIndices[:nTrain], :]
        if nValidate > 0:
            GOESval = goes[rowIndices[nTrain:nTrain+nValidate], :]
        GOEStest = goes[rowIndices[nTrain+nValidate:nTrain+nValidate+nTest], :]

    if raob is not None:
        RAOBtrain = raob[rowIndices[:nTrain], :]
        if nValidate > 0:
            RAOBval = raob[rowIndices[nTrain:nTrain+nValidate], :]
        RAOBtest = raob[rowIndices[nTrain+nValidate:nTrain+nValidate+nTest], :]
        
    if nValidate > 0:
        return (RAPtrain, RAPval, RAPtest,
                RTMAtrain, RTMAval, RTMAtest,
                GOEStrain, GOESval, GOEStest,
                RAOBtrain, RAOBval, RAOBtest)

    return (RAPtrain, RAPtest,
            RTMAtrain, RTMAtest,
            GOEStrain, GOEStest,
            RAOBtrain, RAOBtest)


def partition(X, T, percentages, shuffle=False, seed=1234):
    """Usage: Xtrain,Train,Xvalidate,Tvalidate,Xtest,Ttest = partition(X,T,(0.6,0.2,0.2),shuffle=False,classification=True)
      X is nSamples x nFeatures. 
      percentages can have just two values, for partitioning into train and test only
    """
    validateFraction = 0
    if isinstance(percentages, (tuple, list)):
        trainFraction = percentages[0]
        testFraction = percentages[-1]
        if len(percentages) == 3:
            validateFraction = percentages[1]

    elif isinstance(percentages, float):
        trainFraction = percentages
        testFraction = 1 - trainFraction

    else:
        raise TypeError(
            f'percentages {percentages} must be of the following (train, val, test) or 0.8 for train')

    rowIndices = np.arange(X.shape[0])
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(rowIndices)

    # regression, so do not partition according to targets.
    n = X.shape[0]
    nTrain = round(trainFraction * n)
    nValidate = round(validateFraction * n)
    nTest = round(testFraction * n)
    if nTrain + nValidate + nTest > n:
        nTest = n - nTrain - nValidate
    Xtrain = X[rowIndices[:nTrain], :]
    Ttrain = T[rowIndices[:nTrain], :]
    if nValidate > 0:
        Xvalidate = X[rowIndices[nTrain:nTrain+nValidate], :]
        Tvalidate = T[rowIndices[nTrain:nTrain+nValidate], :]
    Xtest = X[rowIndices[nTrain+nValidate:nTrain+nValidate+nTest], :]
    Ttest = T[rowIndices[nTrain+nValidate:nTrain+nValidate+nTest], :]
        
    if nValidate > 0:
        return Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest
    else:
        return Xtrain, Ttrain, Xtest, Ttest
        
        
######################################################################


def confusion_matrix(actual, predicted, classes):
    nc = len(classes)
    confmat = np.zeros((nc, nc))
    for ri in range(nc):
        trues = (actual == classes[ri]).squeeze()
        predictedThisClass = predicted[trues]
        keep = trues
        predictedThisClassAboveThreshold = predictedThisClass
        for ci in range(nc):
            confmat[ri, ci] = np.sum(
                predictedThisClassAboveThreshold == classes[ci]) / float(np.sum(keep))
    print_confusion_matrix(confmat, classes)
    return confmat


def print_confusion_matrix(confmat, classes):
    print('   ', end='')
    for i in classes:
        print('%5d' % (i), end='')
    print('\n    ', end='')
    print('{:s}'.format('------'*len(classes)))
    for i, t in enumerate(classes):
        print('{:2d} |'.format(t), end='')
        for i1, t1 in enumerate(classes):
            if confmat[i, i1] == 0:
                print('  0  ', end='')
            else:
                print('{:5.1f}'.format(100*confmat[i, i1]), end='')
        print()

######################################################################


def draw(Vs, W, inputNames=None, outputNames=None, gray=False):

    def isOdd(x):
        return x % 2 != 0

    W = Vs + [W]
    nLayers = len(W)

    # calculate xlim and ylim for whole network plot
    #  Assume 4 characters fit between each wire
    #  -0.5 is to leave 0.5 spacing before first wire
    xlim = max(map(len, inputNames))/4.0 if inputNames else 1
    ylim = 0

    for li in range(nLayers):
        ni, no = W[li].shape  # no means number outputs this layer
        if not isOdd(li):
            ylim += ni + 0.5
        else:
            xlim += ni + 0.5

    ni, no = W[nLayers-1].shape
    if isOdd(nLayers):
        xlim += no + 0.5
    else:
        ylim += no + 0.5

    # Add space for output names
    if outputNames:
        if isOdd(nLayers):
            ylim += 0.25
        else:
            xlim += round(max(map(len, outputNames))/4.0)

    ax = plt.gca()

    character_width_factor = 0.07
    padding = 2
    if inputNames:
        x0 = max([1, max(map(len, inputNames)) *
                  (character_width_factor * 3.5)])
    else:
        x0 = 1
    y0 = 0  # to allow for constant input to first layer
    # First Layer
    if inputNames:
        y = 0.55
        for n in inputNames:
            y += 1
            ax.text(x0 - (character_width_factor * padding), y,
                    n, horizontalalignment="right", fontsize=20)

    patches = []
    for li in range(nLayers):
        thisW = W[li]
        maxW = np.max(np.abs(thisW))
        ni, no = thisW.shape
        if not isOdd(li):
            # Even layer index. Vertical layer. Origin is upper left.
            # Constant input
            ax.text(x0-0.2, y0+0.5, '1', fontsize=20)
            for i in range(ni):
                ax.plot((x0, x0+no-0.5), (y0+i+0.5, y0+i+0.5), color='gray')
            # output lines
            for i in range(no):
                ax.plot((x0+1+i-0.5, x0+1+i-0.5), (y0, y0+ni+1), color='gray')
            # cell "bodies"
            xs = x0 + np.arange(no) + 0.5
            ys = np.array([y0+ni+0.5]*no)
            for x, y in zip(xs, ys):
                patches.append(pltpatch.RegularPolygon(
                    (x, y-0.4), 3, 0.3, 0, color='#555555'))
            # weights
            if gray:
                colors = np.array(["black", "gray"])[(thisW.flat >= 0)+0]
            else:
                colors = np.array(["red", "green"])[(thisW.flat >= 0)+0]
            xs = np.arange(no) + x0+0.5
            ys = np.arange(ni) + y0 + 0.5
            coords = np.meshgrid(xs, ys)
            for x, y, w, c in zip(coords[0].flat, coords[1].flat,
                                  np.abs(thisW/maxW).flat, colors):
                patches.append(pltpatch.Rectangle(
                    (x-w/2, y-w/2), w, w, color=c))
            y0 += ni + 1
            x0 += -1  # shift for next layer's constant input
        else:
            # Odd layer index. Horizontal layer. Origin is upper left.
            # Constant input
            ax.text(x0+0.5, y0-0.2, '1', fontsize=20)
            # input lines
            for i in range(ni):
                ax.plot((x0+i+0.5,  x0+i+0.5), (y0, y0+no-0.5), color='gray')
            # output lines
            for i in range(no):
                ax.plot((x0, x0+ni+1), (y0+i+0.5, y0+i+0.5), color='gray')
            # cell "bodies"
            xs = np.array([x0 + ni + 0.5] * no)
            ys = y0 + 0.5 + np.arange(no)
            for x, y in zip(xs, ys):
                patches.append(pltpatch.RegularPolygon(
                    (x-0.4, y), 3, 0.3, -np.pi/2, color='#555555'))
            # weights
            if gray:
                colors = np.array(["black", "gray"])[(thisW.flat >= 0)+0]
            else:
                colors = np.array(["red", "green"])[(thisW.flat >= 0)+0]
            xs = np.arange(ni)+x0 + 0.5
            ys = np.arange(no)+y0 + 0.5
            coords = np.meshgrid(xs, ys)
            for x, y, w, c in zip(coords[0].flat, coords[1].flat,
                                  np.abs(thisW/maxW).flat, colors):
                patches.append(pltpatch.Rectangle(
                    (x-w/2, y-w/2), w, w, color=c))
            x0 += ni + 1
            y0 -= 1  # shift to allow for next layer's constant input

    collection = pltcoll.PatchCollection(patches, match_original=True)
    ax.add_collection(collection)

    # Last layer output labels
    if outputNames:
        if isOdd(nLayers):
            x = x0+1.5
            for n in outputNames:
                x += 1
                ax.text(x, y0+0.5, n, fontsize=20)
        else:
            y = y0+0.6
            for n in outputNames:
                y += 1
                ax.text(x0+0.2, y, n, fontsize=20)
    ax.axis([0, xlim, ylim, 0])
    ax.axis('off')

