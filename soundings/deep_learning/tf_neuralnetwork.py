import copy
import datetime
import time
import random
import pickle

import os
import numpy as np
import tensorflow as tf

from soundings.deep_learning import callbacks
from soundings.deep_learning import tf_metrics as metrics

def loadnn(path):
    try:
        with open(path + '/class.pickle', 'rb') as f:
            c = pickle.load(f)
        tf_model_path = os.path.join(path, 'model.h5')
        c.model = tf.keras.models.load_model(tf_model_path)
        return c
    except Exception as e:
        print(e)
        

class NeuralNetwork():
    def __init__(self, n_inputs, n_hiddens_list, n_outputs, activation='tanh', seed=None):

        if not isinstance(n_hiddens_list, list):
            raise Exception(
                f'{type(self).__name__}: n_hiddens_list must be a list.')

        self.seed = seed
        self._set_seed()
        tf.keras.backend.clear_session()

        self.n_inputs = n_inputs
        self.n_hiddens_list = n_hiddens_list
        self.n_outputs = n_outputs

        X = Z = tf.keras.Input(shape=(n_inputs,))
        if not (n_hiddens_list == [] or n_hiddens_list == [0]):
            for i, units in enumerate(n_hiddens_list):
                Z = tf.keras.layers.Dense(units)(Z)
                Z = tf.keras.layers.BatchNormalization()(Z)                    
                Z = tf.keras.layers.Activation(activation)(Z)
                Z = tf.keras.layers.Dropout(0.2)(Z)
        Y = tf.keras.layers.Dense(n_outputs)(Z)
        self.model = tf.keras.Model(inputs=X, outputs=Y)

        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None

        self.history = None
        self.training_time = None

    def __repr__(self):
        str = f'{type(self).__name__}({self.n_inputs}, {self.n_hiddens_list}, {self.n_outputs})'
        if self.history:
            str += f"\n  Final objective value is {self.history['loss'][-1]:.5f} in {self.training_time:.4f} seconds."
        else:
            str += '  Network is not trained.'
        return str
    
    def _set_seed(self):
        if self.seed:
            np.random.seed(self.seed)
            random.seed(self.seed)
            tf.random.set_seed(self.seed)

    def _setup_standardize(self, X, T):
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xconstant = self.Xstds == 0
            self.XstdsFixed = copy.copy(self.Xstds)
            self.XstdsFixed[self.Xconstant] = 1

        if self.Tmeans is None:
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            self.Tconstant = self.Tstds == 0
            self.TstdsFixed = copy.copy(self.Tstds)
            self.TstdsFixed[self.Tconstant] = 1

    def _standardizeX(self, X):
        result = (X - self.Xmeans) / self.XstdsFixed
        result[:, self.Xconstant] = 0.0
        return result

    def _unstandardizeX(self, Xs):
        return self.Xstds * Xs + self.Xmeans

    def _standardizeT(self, T):
        result = (T - self.Tmeans) / self.TstdsFixed
        result[:, self.Tconstant] = 0.0
        return result

    def _unstandardizeT(self, Ts):
        return self.Tstds * Ts + self.Tmeans

    def train(self, X, T, n_epochs, batch_size, method='sgd',
              verbose=False, learning_rate=0.001, validation=None, loss_f=None):
        """Use Keras Functional API to train model"""

        self._set_seed()
        self._setup_standardize(X, T)
        X = self._standardizeX(X)
        T = self._standardizeT(T)

        if validation is not None:
            try:
                validation = (self._standardizeX(
                    validation[0]), self._standardizeT(validation[1]))
            except:
                raise TypeError(
                    f'validation must be of the following shape: (X, T)')
                
        try:
            if method == 'sgd':
                algo = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            elif method == 'adam':
                algo = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif method == 'Adagrad':
                algo = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
            elif method == 'RMSprop':
                algo = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            elif method == 'Adadelta':
                algo = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
        except:
            raise Exception(
                "train: method={method} not one of 'scg' or 'adam'")

        loss = tf.keras.losses.MSE if loss_f == None else loss_f
        
        self.model.compile(optimizer=algo, loss=loss,
                           metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                    metrics.wrapper_mse(self._unstandardizeT)])
        callback = [callbacks.TrainLogger(n_epochs, step=1)] if verbose else None
        start_time = time.time()
        self.history = self.model.fit(X, T, batch_size=batch_size, epochs=n_epochs,
                                      verbose=0, callbacks=callback,
                                      validation_data=validation).history
        self.training_time = time.time() - start_time
        return self

    def use(self, X):
        # Set to error logging after model is trained
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        X = self._standardizeX(X)
        Y = self._unstandardizeT(self.model.predict(X))
        return Y
    
    def save(self, path):
        tf_model_path = os.path.join(path, 'model.h5')
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(tf_model_path)
        del self.model
        with open(os.path.join(path, 'class.pickle'), 'wb') as f:
            pickle.dump(self, f)
        self.model = tf.keras.models.load_model(tf_model_path)
  
    
class ConvolutionalNeuralNetwork(NeuralNetwork):
    def __init__(self, n_inputs, n_units_in_conv_layers,
                 kernels_size_and_stride, n_outputs, activation='tanh', seed=None):

        if not isinstance(n_units_in_conv_layers, (list, tuple)):
            raise Exception(
                f'{type(self).__name__}: n_units_in_conv_layers must be a list.')

        if not isinstance(kernels_size_and_stride, list):
            raise Exception(
                f'{type(self).__name__}: kernels_size_and_stride must be a list.')

        self.seed = seed
        self._set_seed()
        tf.keras.backend.clear_session()

        self.n_inputs = n_inputs
        self.n_units_in_conv_layers = n_units_in_conv_layers
        self.kernels_size_and_stride = kernels_size_and_stride
        self.n_outputs = n_outputs

        X = tf.keras.Input(shape=n_inputs)
        Z = X
        i = 0
        for (kernel, stride), units in zip(kernels_size_and_stride, n_units_in_conv_layers):
            Z = tf.keras.layers.Conv1D(
                units, kernel_size=kernel, strides=stride, activation=activation, padding='same')(Z)
            Z = tf.keras.layers.MaxPooling1D(pool_size=2)(Z)
            if i % 2 == 0:
                Z = tf.keras.layers.Dropout(0.20)(Z)
            i += 1
        Y = tf.keras.layers.Dense(n_outputs)(tf.keras.layers.Flatten()(Z))

        self.model = tf.keras.Model(inputs=X, outputs=Y)

        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None

        self.history = None
        self.training_time = None

    def __repr__(self):
        str = f'{type(self).__name__}({self.n_inputs}, {self.n_units_in_conv_layers}, {self.kernels_size_and_stride}, {self.n_outputs})'
        if self.history:
            str += f"\n  Final objective value is {self.history['loss'][-1]:.5f} in {self.training_time:.4f} seconds."
        else:
            str += '  Network is not trained.'
        return str


class ConvolutionalAutoEncoder(NeuralNetwork):
    def __init__(self, n_inputs, n_units_in_conv_layers,
                 kernels_size_and_stride, n_outputs, activation='relu', n_hidden_dims=100, seed=None):

        if not isinstance(n_units_in_conv_layers, (list, tuple)):
            raise Exception(
                f'{type(self).__name__}: n_units_in_conv_layers must be a list.')

        if not isinstance(kernels_size_and_stride, list):
            raise Exception(
                f'{type(self).__name__}: kernels_size_and_stride must be a list.')

        self.seed = seed
        self._set_seed()
        tf.keras.backend.clear_session()

        self.n_inputs = n_inputs
        self.n_units_in_conv_layers = n_units_in_conv_layers
        self.kernels_size_and_stride = kernels_size_and_stride
        self.n_outputs = n_outputs
        self.n_hidden_dim = n_hidden_dims

        # encoder
        X = tf.keras.Input(shape=n_inputs)
        Z = X
        for (kernel, stride), units in zip(kernels_size_and_stride, n_units_in_conv_layers):
            Z = tf.keras.layers.Conv1D(
                units, kernel_size=kernel, strides=stride, activation=activation, padding='same')(Z)
            Z = tf.keras.layers.MaxPooling1D(pool_size=2)(Z)

        # latent vector
        conv_shape = Z.shape[1:]
        F = tf.keras.layers.Flatten()(Z)
        Z = tf.keras.layers.Dense(n_hidden_dims, activation='tanh')(F)

        # decoder (input of `n_hidden_dim`)
        Z = tf.keras.layers.Dense(F.shape[1], activation='tanh')(Z)
        Z = tf.keras.layers.Reshape(conv_shape)(Z)

        for (kernel, stride), units in zip(reversed(kernels_size_and_stride), reversed(n_units_in_conv_layers)):
            Z = tf.keras.layers.Conv1D(
                units, kernel_size=kernel, strides=stride, activation=activation, padding='same')(Z)
            Z = tf.keras.layers.UpSampling1D(size=2)(Z)
                  
        Z = tf.keras.layers.Conv1D(
            1, kernel_size=kernels_size_and_stride[0][0], strides=kernels_size_and_stride[0][1],
            activation=activation, padding='same')(Z)      
        # Y = tf.keras.layers.Flatten()(Z)
        Y = tf.keras.layers.Dense(n_inputs[0])(tf.keras.layers.Flatten()(Z))
        self.model = tf.keras.Model(inputs=X, outputs=Y)

        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None

        self.history = None
        self.training_time = None

    def __repr__(self):
        str = f'{type(self).__name__}({self.n_inputs}, {self.n_units_in_conv_layers}, {self.kernels_size_and_stride}, {self.n_outputs})'
        if self.history:
            str += f"\n  Final objective value is {self.history['loss'][-1]:.5f} in {self.training_time:.4f} seconds."
        else:
            str += '  Network is not trained.'
        return str
    
    
    
class SkipNeuralNetwork(NeuralNetwork):
    def __init__(self, n_inputs, n_units_in_conv_layers,
                 kernels_size_and_stride, n_outputs, activation='relu', n_hidden_dims=100, seed=None):

        if not isinstance(n_units_in_conv_layers, (list, tuple)):
            raise Exception(
                f'{type(self).__name__}: n_units_in_conv_layers must be a list.')

        if not isinstance(kernels_size_and_stride, list):
            raise Exception(
                f'{type(self).__name__}: kernels_size_and_stride must be a list.')

        self.seed = seed
        self._set_seed()
        tf.keras.backend.clear_session()

        self.n_inputs = n_inputs
        self.n_units_in_conv_layers = n_units_in_conv_layers
        self.kernels_size_and_stride = kernels_size_and_stride
        self.n_outputs = n_outputs
        self.n_hidden_dim = n_hidden_dims

        # encoder
        X = Z = tf.keras.Input(shape=n_inputs)
        
        for (kernel, stride), units in zip(kernels_size_and_stride, n_units_in_conv_layers):
            Z = tf.keras.layers.Conv1D(
                units, kernel_size=kernel, strides=stride, padding='same')(Z)
            Z = tf.keras.layers.BatchNormalization()(Z)                    
            Z = tf.keras.layers.Activation(activation)(Z)
            Z = tf.keras.layers.MaxPooling1D(pool_size=2)(Z)
       
        encoder = tf.keras.Model(X, Z)
        skips = list(reversed([layer for layer in encoder.layers if 'max' in layer.name]))
        # latent vector
        conv_shape = Z.shape[1:]
        F = tf.keras.layers.Flatten()(Z)
        Z = tf.keras.layers.Dropout(0.50)(Z)
        Z = tf.keras.layers.Dense(n_hidden_dims, activation='tanh')(F)
        
        # decoder (input of `n_hidden_dim`)
        Z = tf.keras.layers.Dropout(0.50)(Z)
        Z = tf.keras.layers.Dense(F.shape[1], activation='tanh')(Z)
        Z = tf.keras.layers.Reshape(conv_shape)(Z)
        for (kernel, stride), units, skip in zip(reversed(kernels_size_and_stride),
                                                 reversed(n_units_in_conv_layers),
                                                 skips):
            # print(Z.shape, skip.output.shape)
            Z = tf.keras.layers.Concatenate()([Z, skip.output])
            Z = tf.keras.layers.Conv1D(
                units, kernel_size=kernel, strides=stride, padding='same')(Z)
            Z = tf.keras.layers.BatchNormalization()(Z)                    
            Z = tf.keras.layers.Activation(activation)(Z)
            Z = tf.keras.layers.UpSampling1D(size=2)(Z)
            
        Z = tf.keras.layers.Conv1D(
            1, kernel_size=kernels_size_and_stride[0][0], strides=kernels_size_and_stride[0][1],
            activation=activation, padding='same')(Z)      

        Y = tf.keras.layers.Dense(n_inputs[0])(tf.keras.layers.Flatten()(Z))
        self.model = tf.keras.Model(inputs=X, outputs=Y)

        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None

        self.history = None
        self.training_time = None

    def __repr__(self):
        str = f'{type(self).__name__}({self.n_inputs}, {self.n_units_in_conv_layers}, {self.kernels_size_and_stride}, {self.n_outputs})'
        if self.history:
            str += f"\n  Final objective value is {self.history['loss'][-1]:.5f} in {self.training_time:.4f} seconds."
        else:
            str += '  Network is not trained.'
        return str
    
    

class MultiNeuralNetwork():
    def __init__(self, n_im_inputs, n_rap_inputs, im_hiddens_list, 
                 n_units_in_conv_layers, kernels_size_and_stride,
                 n_outputs, im_activation='tanh', rap_activation='relu', seed=None):
        
        assert ((len(n_im_inputs) == 3)), f'Image must be HxWxC dimensions, {n_im_inputs}'
        assert (isinstance(im_hiddens_list, list)), f'{type(self).__name__}: im_hiddens_list must be a list.'
        assert (isinstance(n_units_in_conv_layers, list)), f'{type(self).__name__}: n_units_in_conv_layers must be a list.'
        assert (isinstance(kernels_size_and_stride, list)), f'{type(self).__name__}: kernels_size_and_stride must be a list.'
        
        self.seed = seed
        self._set_seed()
        tf.keras.backend.clear_session()

        self.n_im_inputs = n_im_inputs
        self.n_rap_inputs = n_rap_inputs
        self.im_hiddens_list = im_hiddens_list
        self.n_units_in_conv_layers = n_units_in_conv_layers
        self.kernels_size_and_stride = kernels_size_and_stride
        self.n_outputs = n_outputs
        
        X1 = tf.keras.Input(shape=n_im_inputs, name='im')
        
        Z = tf.keras.layers.Flatten()(X1)
        for units in im_hiddens_list[:-1]:
            Z = tf.keras.layers.Dense(units, activation=im_activation)(Z)
            
        Z = tf.keras.layers.Dense(im_hiddens_list[-1])(Z)
        
        def _pad_upper_zeros(z):
            """Pad `n_rap_inputs` minus the previous unit outputs with zeros to every 
            tensor in the batch. This isolates the information only near the surface.
            """
            prev_units = n_rap_inputs[0] - z.shape[1]
            zeros =  tf.expand_dims(tf.zeros(prev_units), axis=0)
            repeat = tf.squeeze(tf.keras.layers.RepeatVector(tf.shape(z)[0])(zeros), axis=0)
            return tf.expand_dims(tf.concat([z, repeat], axis=1), axis=-1)
     
        # pad with zeros to match rap input
        out = tf.keras.layers.Lambda(_pad_upper_zeros)(Z)
        # out = tf.expand_dims(tf.keras.layers.Dense(256)(Z), axis=-1)
        
        X2 = tf.keras.Input(shape=n_rap_inputs, name='rap')
        Z = tf.keras.layers.Concatenate(axis=2)([X2, out])

        for (kernel, stride), units in zip(kernels_size_and_stride, n_units_in_conv_layers):
            Z = tf.keras.layers.Conv1D(units, kernel_size=kernel, strides=stride,
                                       activation=rap_activation, padding='same')(Z)
            Z = tf.keras.layers.MaxPooling1D(pool_size=2)(Z)
        
        Y = tf.keras.layers.Dense(n_outputs, name='out')(tf.keras.layers.Flatten()(Z))
        self.model = tf.keras.Model(inputs=[X1, X2], outputs=Y)
        
        self.IMmeans = None
        self.IMstds = None
        self.RAPmeans = None
        self.RAPstds = None
        self.RAOBmeans = None
        self.RAOBstds = None
        
        self.history = None
        self.training_time = None

    def __repr__(self):
        str = f'{type(self).__name__}({self.n_outputs})'
        if self.history:
            str += f"\n  Final objective value is {self.history['loss'][-1]:.5f} in {self.training_time:.4f} seconds."
        else:
            str += '  Network is not trained.'
        return str
    
    def _set_seed(self):
        if self.seed:
            np.random.seed(self.seed)
            random.seed(self.seed)
            tf.random.set_seed(self.seed)
            
    def _setup_standardize(self, im, rap, raob):
        if self.IMmeans is None:
            self.IMmeans = im.mean(axis=0)
            self.IMstds = im.std(axis=0)
            self.IMconstant = self.IMstds == 0
            self.IMstdsFixed = copy.copy(self.IMstds)
            self.IMstdsFixed[self.IMconstant] = 1
       
        if self.RAPmeans is None:
            self.RAPmeans = rap.mean(axis=0)
            self.RAPstds = rap.std(axis=0)
            self.RAPconstant = self.RAPstds == 0
            self.RAPstdsFixed = copy.copy(self.RAPstds)
            self.RAPstdsFixed[self.RAPconstant] = 1

        if self.RAOBmeans is None:
            self.RAOBmeans = raob.mean(axis=0)
            self.RAOBstds = raob.std(axis=0)
            self.RAOBconstant = self.RAOBstds == 0
            self.RAOBstdsFixed = copy.copy(self.RAOBstds)
            self.RAOBstdsFixed[self.RAOBconstant] = 1

    def _standardizeIM(self, im):
        result = (im - self.IMmeans) / self.IMstdsFixed
        result[:, self.IMconstant] = 0.0
        return result

    def _unstandardizeIM(self, im):
        return self.IMstds * im + self.IMmeans
            
    def _standardizeRAP(self, rap):
        result = (rap - self.RAPmeans) / self.RAPstdsFixed
        result[:, self.RAPconstant] = 0.0
        return result

    def _unstandardizeRAP(self, rap):
        return self.RAPstds * rap + self.RAPmeans

    def _standardizeRAOB(self, raob):
        result = (raob - self.RAOBmeans) / self.RAOBstdsFixed
        result[:, self.RAOBconstant] = 0.0
        return result

    def _unstandardizeRAOB(self, raob):
        return self.RAOBstds * raob + self.RAOBmeans
    
    def train(self, im, rap, raob, n_epochs, batch_size, method='sgd',
              verbose=False, learning_rate=0.001, validation=None, loss_f=None):
        """Use Keras Functional API to train model"""
        
        self._set_seed()
        self._setup_standardize(im, rap, raob)
        
        im = self._standardizeIM(im)
        rap  = self._standardizeRAP(rap)
        raob = self._standardizeRAOB(raob)
        
        if validation is not None:
            try:
                # TODO: Implement this
                validation = None
            except:
                raise TypeError(
                    f'validation must be of the following shape: (im, rap, raob)')

        try:
            if method == 'sgd':
                algo = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            elif method == 'adam':
                algo = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        except:
            raise Exception(
                "train: method={method} not one of 'scg' or 'adam'")

        loss = tf.keras.losses.MSE if loss_f == None else loss_f
        self.model.compile(optimizer=algo, loss=loss,
                           metrics=[tf.keras.metrics.RootMeanSquaredError()])

        callback = [callbacks.TrainLogger(n_epochs, step=5)] if verbose else None
        start_time = time.time()
        self.history = self.model.fit({'im': im, 'rap': rap}, {'out': raob},
                                      batch_size=batch_size, epochs=n_epochs, verbose=0,
                                      callbacks=callback, validation_data=validation).history
        self.training_time = time.time() - start_time
        return self

    def use(self, X):
        """
        Inputs:
            X : {'im': im, 'rap': rap}
        """
        # Set to error logging after model is trained
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        im = self._standardizeIM(X['im'])
        rap  = self._standardizeRAP(X['rap'])
        Y = self._unstandardizeRAOB(self.model.predict({'im': im, 'rap': rap}))
        return Y
        
    def save(self, path):
        self.model.save(path)
        del self.model
        with open(path + '/class.pickle', 'wb') as f:
            pickle.dump(self, f)
        self.model = tf.keras.models.load_model(path)