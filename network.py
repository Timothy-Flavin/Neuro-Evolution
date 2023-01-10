import numpy as np

def sigmoid(num):
  return 1.0/(1+np.exp(-num))

def relu(num):
  return np.maximum(num,0)

class network:
  def __init__(self, input_size=18, layer_sizes=np.array([32, 16, 9]), activations=["relu","relu","sigmoid"]):
    
    if layer_sizes.shape[0] != len(activations):
      raise(ValueError("Layer sizes and activations have different numbers of layers"))

    self.act_names = activations
    self.acts = []
    for a in activations:
      if a == "sigmoid":
        self.acts.append(sigmoid)
      elif a == "relu":
        self.acts.append(relu)
      else:
        raise(Exception("Activations must be either 'relu' or 'sigmoid'"))
    self.layer_sizes = layer_sizes
    self.input_size = input_size

    self.weights = []
    self.biases = []
    self.weights.append(np.random.random((input_size, layer_sizes[0]))-0.5)
    self.biases.append(np.random.random(size=self.layer_sizes[0])-0.5)
    for i in range(1,len(self.acts)):
      self.weights.append(np.random.random((layer_sizes[i-1],layer_sizes[i]))-0.5)
      self.biases.append(np.random.random(size=self.layer_sizes[i])-0.5)

  def print_self(self, verbose=False):
    print(f"\nInput size: {self.input_size}\nLayer sizes: {self.layer_sizes},\nActivations: {self.act_names}")
    
    if verbose:
      for i in range(len(self.act_names)):
        print(f"\nWeights [{i}]: ")
        print(self.weights[i])
        print(f"\nBiases [{i}]: ")
        print(self.biases[i])

  def feed_forward(self, input_data, verbose = False):
    mat = input_data
    if verbose:
      print(mat)
    for i in range(len(self.act_names)):
      if verbose:
        print(mat.shape)
        print(self.weights[i].shape)
      mat = np.matmul(mat,self.weights[i])
      if verbose:      
        print(f"After matmul [{i}]")
        print(mat)
      mat += self.biases[i]
      if verbose:      
        print(f"After biases [{i}]")
        print(mat)
      mat = self.acts[i](mat)
      if verbose:      
        print(f"After activation [{i}]")
        print(mat)
    return mat
  
  def move(self, state):
    mat = self.feed_forward(state)
    return np.argmax(mat)