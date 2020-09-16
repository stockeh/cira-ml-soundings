import tensorflow as tf

def wrapper_truncated_mse(_unstandardizeT, sfc_boundary=25):
    def truncated_mse(y_pred, y_true):
        y_pred = _unstandardizeT(y_pred)[:,:sfc_boundary]
        y_true = _unstandardizeT(y_true)[:,:sfc_boundary]
        return tf.math.reduce_mean(tf.square(y_pred - y_true))
    return truncated_mse


def wrapper_mse(_unstandardizeT):
    def mse(y_pred, y_true):
        y_pred = _unstandardizeT(y_pred)
        y_true = _unstandardizeT(y_true)
        return tf.math.reduce_mean(tf.square(y_pred - y_true))
    return mse
    
    
def unstd_rmse(_unstandardizeT):
    """NOTE: This does not work as expected.
    https://stackoverflow.com/questions/62115817/
    tensorflow-keras-rmse-metric-returns-different-results-than-my-own-built-rmse-lo
    """
    def rmse(y_pred, y_true):
        y_pred = _unstandardizeT(y_pred)
        y_true = _unstandardizeT(y_true)
        return tf.math.sqrt(tf.math.reduce_mean(tf.square(y_pred - y_true)))
    return rmse
    