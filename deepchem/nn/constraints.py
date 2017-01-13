from __future__ import absolute_import
from keras import backend as K
from .activations import get_from_module

class Constraint(object):

  def __call__(self, p):
    return p

class MaxNorm(Constraint):
  """MaxNorm weight constraint.

  Constrains the weights incident to each hidden unit
  to have a norm less than or equal to a desired value.

  # Arguments
    m: the maximum norm for the incoming weights.
    axis: integer, axis along which to calculate weight norms.
        For instance, in a `Dense` layer the weight matrix
        has shape `(input_dim, output_dim)`,
        set `axis` to `0` to constrain each weight vector
        of length `(input_dim,)`.
        In a `Convolution2D` layer with `dim_ordering="tf"`,
        the weight tensor has shape
        `(rows, cols, input_depth, output_depth)`,
        set `axis` to `[0, 1, 2]`
        to constrain the weights of each filter tensor of size
        `(rows, cols, input_depth)`.

  # References
    - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
  """

  def __init__(self, m=2, axis=0):
    self.m = m
    self.axis = axis

  def __call__(self, p):
    norms = K.sqrt(K.sum(K.square(p), axis=self.axis, keepdims=True))
    desired = K.clip(norms, 0, self.m)
    p *= (desired / (K.epsilon() + norms))
    return p


class NonNeg(Constraint):
  """Constrains the weights to be non-negative.
  """

  def __call__(self, p):
    p *= tf.cast(p >= 0., tf.float32)
    return p


class UnitNorm(Constraint):
  """Constrains the weights incident to each hidden unit to have unit norm.

  # Arguments
      axis: integer, axis along which to calculate weight norms.
          For instance, in a `Dense` layer the weight matrix
          has shape `(input_dim, output_dim)`,
          set `axis` to `0` to constrain each weight vector
          of length `(input_dim,)`.
          In a `Convolution2D` layer with `dim_ordering="tf"`,
          the weight tensor has shape
          `(rows, cols, input_depth, output_depth)`,
          set `axis` to `[0, 1, 2]`
          to constrain the weights of each filter tensor of size
          `(rows, cols, input_depth)`.
  """

  def __init__(self, axis=0):
    self.axis = axis

  def __call__(self, p):
    return p / (1e-7 + K.sqrt(K.sum(K.square(p),
                                    axis=self.axis,
                                    keepdims=True)))

# Aliases.

maxnorm = MaxNorm
nonneg = NonNeg
unitnorm = UnitNorm


def get(identifier, kwargs=None):
  return get_from_module(identifier, globals(), 'constraint',
                         instantiate=True, kwargs=kwargs)
