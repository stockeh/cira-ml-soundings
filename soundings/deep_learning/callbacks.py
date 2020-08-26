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