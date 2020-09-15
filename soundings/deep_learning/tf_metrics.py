import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import util as tf_losses_utils


class RMSE(tf.keras.metrics.Mean):
    """ Implements the same logic as `tf.keras.metrics.RootMeanSquaredError`,
    but unstandardizes the outputs before computing to maintain original units.
    """
    def __init__(self, _unstandardizeT, name='rmse', dtype=None):
        super(RMSE, self).__init__(name, dtype=dtype)
        self._unstandardizeT = _unstandardizeT
        
    def update_state(self, T, Y, sample_weight=None):
        """Accumulates **unstandardized** root mean squared error statistics.
        
        args:
        ---
          T: The ground truth values.
          Y: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`. 
        returns:
        ---
          Update op.
        """
        T = math_ops.cast(self._unstandardizeT(T), self._dtype)
        Y = math_ops.cast(self._unstandardizeT(Y), self._dtype)
        Y, T = tf_losses_utils.squeeze_or_expand_dimensions(Y, T)
        error_sq = math_ops.squared_difference(Y, T)
        return super(RMSE, self).update_state(error_sq, sample_weight=sample_weight)
    
    def result(self):
        return math_ops.sqrt(math_ops.div_no_nan(self.total, self.count))
    
    
class SurfaceRMSE(tf.keras.metrics.Mean):
    """ Implements the same logic as `tf.keras.metrics.RootMeanSquaredError`,
    but unstandardizes the outputs before computing to maintain original units.
    
    The 
    np.argmin(abs(raob[:,:,ALTITUDE] - 2000))
    
    """
    def __init__(self, _unstandardizeT, name='sfc_rmse', dtype=None):
        super(SurfaceRMSE, self).__init__(name, dtype=dtype)
        self._unstandardizeT = _unstandardizeT
        self.surface_boundary_index = 25
        
    def update_state(self, T, Y, sample_weight=None):
        """Accumulates **unstandardized** root mean squared error statistics.
        
        args:
        ---
          T: The ground truth values.
          Y: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`. 
        returns:
        ---
          Update op.
        """
        T = math_ops.cast(self._unstandardizeT(T)[:, :self.surface_boundary_index], self._dtype)
        Y = math_ops.cast(self._unstandardizeT(Y)[:, :self.surface_boundary_index], self._dtype)
        Y, T = tf_losses_utils.squeeze_or_expand_dimensions(Y, T)
        error_sq = math_ops.squared_difference(Y, T)
        return super(SurfaceRMSE, self).update_state(error_sq, sample_weight=sample_weight)
    
    def result(self):
        return math_ops.sqrt(math_ops.div_no_nan(self.total, self.count))
    
    
    
    
    
    