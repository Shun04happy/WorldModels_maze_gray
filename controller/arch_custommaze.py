import numpy as np
from collections import namedtuple

TIME_FACTOR = 0
NOISE_BIAS = 0
OUTPUT_NOISE = [False, False, False, False]
OUTPUT_SIZE = 4
t=0.8
# def activations(a):
#   a = np.tanh(a)
#   a[1] = (a[1] + 1) / 2
#   a[2] = (a[2] + 1) / 2
#   return a

def activations(action_probs):#softmax
  c = np.max(action_probs)
  exp_action_probs = np.exp(action_probs-c)
  action_probs_soft = exp_action_probs / np.sum(exp_action_probs)
  return action_probs_soft
  
# def activations(action_probs):#Boltzmann
#   T=1/(np.log(t+0.1))
#   c = np.max(action_probs/)
#   exp_action_probs = np.exp(action_probs-c)
#   action_probs_soft = exp_action_probs / np.sum(exp_action_probs)
#   return action_probs_soft
  
class Controller():
    def __init__(self):
        self.time_factor = TIME_FACTOR
        self.noise_bias = NOISE_BIAS
        self.output_noise=OUTPUT_NOISE
        self.activations=activations
        self.output_size = OUTPUT_SIZE
        
    
    



