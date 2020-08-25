import tensorflow as tf
import numpy as np
from soundings.deep_learning import tf_neuralnetwork as nn
from soundings.deep_learning import mlutilities as ml


def setup():
    gpus = tf.config.get_visible_devices('GPU')
    for device in gpus:
        tf.config.experimental.set_memory_growth(device, True)
    
def train(train_percentage, test_percentage, Xtrain, Ttrain, Xtest, Ttest, layer, best_nnet, top):
    setup()
    nnet = nn.ConvolutionalNeuralNetwork(Xtrain.shape[1:], layer, [(10, 1)]*4,
                                         Ttrain.shape[1], activation='relu', seed=1234)
    nnet.train(Xtrain, Ttrain, 20, 16, method='adam', verbose=False, learning_rate=0.001)

    Y = nnet.use(Xtrain)
    train_percentage.append(ml.rmse(Ttrain, Y))

    Y = nnet.use(Xtest)
    temp = ml.rmse(Ttest, Y)
    test_percentage.append(temp)

    if temp < top:
        best_nnet = nnet
        top = temp
        
    return best_nnet, top