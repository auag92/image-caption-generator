import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Utilitites related to gumbel softmax:
def sample_gumbel(trng, shape, eps=np.float32(1e-8), U=None):
  """Sample from Gumbel(0, 1)"""
  if U == None:
    U =torch.empty(shape).uniform_(0., 1.); 
  return -torch.log(-torch.log(U + eps) + eps)

# Utilitites related to gumbel softmax:
def gumbel_softmax_sample(trng, logits, tau, U=None, hard=False):
  """Sample from Gumbel(0, 1)"""
  ylog = logits + sample_gumbel(trng, logits.shape, U=U)
  y = nn.softmax(ylog/tau)

  if hard:
      print('Using hard gumbel')
      # Still working on this
      #one_hot = tensor.cast( tensor.eq(y, y.max(axis=-1,keepdims=1)) ,dtype=config.floatX)
      #y = theano.gradient.disconnected_grad(one_hot -y) + y
  return y

# Utilitites related to gumbel softmax:
def sample_gumbel_np(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = np.random.uniform(0.0,1.0,size=shape)
  return -np.log(-np.log(U + eps) + eps)

# Utilitites related to gumbel softmax:
def gumbel_softmax_sample_np(logits, tau):
  """Sample from Gumbel(0, 1)"""
  y = logits + sample_gumbel_np(logits.shape)
  return nn.softmax(y/tau)

