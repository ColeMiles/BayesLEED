import numpy as np
import os
from GPyOpt.util.general import reshape
from .run_LEED_reduced_dim import Run_LEED_reduced_dim

class LaNiO3_LEED_IV_objective_reduced_dim:
   '''
   Test objective for LEED-IV optimization on LaNiO3.
   We start by only allowing for variation in the z-direction
   '''
   def __init__(self,bounds=None):
      self.input_dim = 8
      if bounds is  None: self.bounds = [(-0.25,+0.25),(-0.25,+0.25),(-0.25,+0.25),(-0.25,+0.25),(-0.25,+0.25),(-0.25,+0.25),(-0.25,0.0),(-0.25,0.0)]
      else: self.bounds = bounds
      self.name = 'LaNiO3_LEED'

   def f(self,X):
      X = reshape(X,self.input_dim)
      n = X.shape[0] # batch size
      if n > 1:
          return 'Error: batch size must be equal to 1'
      if X.shape[1] != self.input_dim:
          return 'Wrong input dimension'
      else:
          # Run LEED code
          pid = str(os.getpid())
          coord_perturbs = X.reshape(self.input_dim)
          rfactor = Run_LEED_reduced_dim(coord_perturbs,pid)
          fval = np.array([rfactor])
          return fval.reshape(n,1)
