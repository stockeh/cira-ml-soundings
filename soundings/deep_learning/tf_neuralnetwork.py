import copy
import datetime
import time
import random

import numpy as np
import tensorflow as tf


class TrainLogger(tf.keras.callbacks.Callback):

    def __init__(self, n_epochs, step=10):
        self.step = step
        self.n_epochs = n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.step == 0:
            print(f"epoch: {epoch}, loss: {logs['loss']:7.5f}")
        elif epoch + 1 == self.n_epochs:
            print(f"epoch: {epoch}, loss: {logs['loss']:7.5f}")
            print('finished!')


class NeuralNetwork():
    def __init__(self, n_inputs, n_hiddens_list, n_outputs, activation='tanh', seed=None):

        if not isinstance(n_hiddens_list, list):
            raise Exception(
                f'{type(self).__name__}: n_hiddens_list must be a list.')

        if seed:
            self.seed = seed
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
                  
        tf.keras.backend.clear_session()

        self.n_inputs = n_inputs
        self.n_hiddens_list = n_hiddens_list
        self.n_outputs = n_outputs

        X = tf.keras.Input(shape=(n_inputs,))
        Z = X
        if not (n_hiddens_list == [] or n_hiddens_list == [0]):
            for i, units in enumerate(n_hiddens_list):
                Z = tf.keras.layers.Dense(units, activation=activation)(Z)
                if i % 2 == 0:
                    Z = tf.keras.layers.BatchNormalization()(Z)
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
            algo = [tf.keras.optimizers.SGD, tf.keras.optimizers.Adam][[
                'sgd', 'adam'].index(method)]
        except:
            raise Exception(
                "train: method={method} not one of 'scg' or 'adam'")

        loss = tf.keras.losses.MSE if loss_f == None else loss_f
        self.model.compile(optimizer=algo(learning_rate), loss=loss,
                           metrics=[tf.keras.metrics.RootMeanSquaredError()])

        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # , tf.keras.callbacks.TensorBoard(histogram_freq=1)
        callback = [TrainLogger(n_epochs, step=5)] if verbose else None

        start_time = time.time()
        self.history = self.model.fit(X, T, batch_size=batch_size, epochs=n_epochs,
                                      verbose=0, callbacks=callback,
                                      validation_data=validation).history
        self.training_time = time.time() - start_time
        return self

    def use(self, X):
        X = self._standardizeX(X)
        Y = self._unstandardizeT(self.model.predict(X))
        return Y


class ConvolutionalNeuralNetwork(NeuralNetwork):
    def __init__(self, n_inputs, n_units_in_conv_layers,
                 kernels_size_and_stride, n_outputs, activation='relu', seed=None):

        if not isinstance(n_units_in_conv_layers, (list, tuple)):
            raise Exception(
                f'{type(self).__name__}: n_units_in_conv_layers must be a list.')

        if not isinstance(kernels_size_and_stride, list):
            raise Exception(
                f'{type(self).__name__}: kernels_size_and_stride must be a list.')

        if seed:
            self.seed = seed
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
                  
        tf.keras.backend.clear_session()

        self.n_inputs = n_inputs
        self.n_units_in_conv_layers = n_units_in_conv_layers
        self.kernels_size_and_stride = kernels_size_and_stride
        self.n_outputs = n_outputs

        X = tf.keras.Input(shape=n_inputs)
        Z = X
        for (kernel, stride), units in zip(kernels_size_and_stride, n_units_in_conv_layers):
            Z = tf.keras.layers.Conv1D(
                units, kernel_size=kernel, strides=stride,  activation=activation, padding='same')(Z)
            Z = tf.keras.layers.MaxPooling1D(pool_size=2)(Z)
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

        if seed:
            self.seed = seed
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
                  
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
