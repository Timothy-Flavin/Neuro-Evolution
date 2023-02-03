import numpy as np

def sigmoid(num):
  return 1.0/(1+np.exp(-num))

def relu(num):
  return np.maximum(num,0)

class network:
  def __init__(self, input_size=18, layer_sizes=np.array([32, 16, 9]), activations=["relu","relu","sigmoid"]):
    self.fitness=0
    self.move = self.move_max
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
  
  def set_move_type(self, mtype = "max"):
    if mtype == "max":
      self.move = self.move_max
    elif mtype == "prob":
      self.move = self.move_prob
    if mtype == "max_legal":
      self.move = self.move_max_legal
    elif mtype == "prob_legal":
      self.move = self.move_prob_legal


  def move_max(self, state, verbose=False):
    mat = self.feed_forward(state)
    if verbose:
      print(f"move type: prob, max: {mat}")
    return np.argmax(mat)
  
  def move_prob(self, state, verbose=True):
    mat = self.feed_forward(state)
    mat /= np.sum(mat)
    mv=0
    if verbose:
      sample = np.random.multinomial(1, mat)
      mv = np.argmax(sample)
      print(f"sample: {sample}")
      print(f"move type: prob, move: {mv}, from probs: {mat}")
      return mv
    else:
      return np.argmax(np.random.multinomial(1, mat))
  
  def move_max_legal(self, state, verbose=False):
    mat = self.feed_forward(state)
    mat=mat/sum(mat)
    if verbose:
      print(f"mat before legal: {mat}")
    inv_state = 1-np.maximum(state[0:9],state[9:])
    mat = mat * inv_state
    mat=mat/sum(mat)
    if verbose:
      print(f"move type: max_legal, max: {mat}")
    return np.argmax(mat)
  
  def move_prob_legal(self, state, verbose=True):
    mat = self.feed_forward(state)
    mat /= np.sum(mat)
    inv_state = 1-np.maximum(state[0:9],state[9:])
    mat = mat * inv_state
    mat=mat/sum(mat)
    mv=0
    if verbose:
      sample = np.random.multinomial(1, mat)
      mv = np.argmax(sample)
      print(f"sample: {sample}")
      print(f"move type: prob_legal, move: {mv}, from probs: {mat}")
      return mv
    else:
      return np.argmax(np.random.multinomial(1, mat))