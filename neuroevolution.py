import numpy as np
from network import network
import random

class rand_legal:
  def move(self, board, verbose=False):
    mv=0
    legal = False
    while not legal:
      mv = random.randint(0,8)
      if board[mv]==0 and board[mv+9]==0:
        legal=True
    return mv

class NeuralGA:
  def default_fitness(self, game, players):
    return game.play(players, print_board = False, verbose = False)

  def __init__(self, pop_size, env, fit_func = default_fitness, input_size=18, layer_sizes=np.array([32, 16, 9]), activations=["relu","relu","sigmoid"], legal_reward=0.1, illegal_reward=-1,win_reward=1):
    self.pop_size = pop_size
    self.population = []
    self.pop_fits = np.zeros(pop_size)
    self.n_plays = np.zeros(pop_size)
    for i in range(pop_size):
      self.population.append(network(input_size, layer_sizes, activations))
    self.fit_func = fit_func
    self.env = env
    self.rand_player = rand_legal()
    self.legal_reward = legal_reward
    self.illegal_reward = illegal_reward
    self.win_reward = win_reward
  
  # eps is the probability of going with the best move instead of sampleing move dist
  def get_fitness_ttt(self, num_plays, epsilon=0.5, fit_laplace=0.01, verbose=0, eps=0.5):
    # for each player, get their fittness
    self.pop_fits = np.zeros(self.pop_fits.shape)
    self.n_plays = np.zeros(self.n_plays.shape)
    for i in range(self.pop_size):
      if verbose>0:
        print(f"Playing games for chromosome[{i}]")
      # play num_plays games and get average return
      p2i=-1
      for pnum in range(num_plays):
        p2i=-1
        opponent = self.rand_player
        # choose neural opponent
        if random.random() < epsilon:
          p2i = random.randint(0,self.pop_size-1)
          opponent = self.population[p2i]
          if verbose>0:
            print(f"Game {pnum} between p1: {i} and p2: {p2i} ", end='')
        # random legal moves opponent
        else:
          if verbose>0:
            print(f"Game {pnum} between p1: {i} and random ", end='')
        
        p1 = self.population[i]
        p2 = opponent
        first = random.randint(0,1)
        reward = 0
        v = verbose>1
        if first == 0:
          if verbose>0:
            print("Player 1 goes first: ")
          reward,reward2 = self.env.play(players=[p1, p2], print_board = v, verbose=v, legal_reward=self.legal_reward, illegal_reward=self.illegal_reward,win_reward=self.win_reward)
        else:
          if verbose>0:
            print("Player 2 goes first: ")
          reward2,reward = self.env.play(players=[p2, p1], print_board = v, verbose=v, legal_reward=self.legal_reward, illegal_reward=self.illegal_reward,win_reward=self.win_reward)
        if verbose>0:
          print(f" outcome: {reward}")
        self.pop_fits[i] += reward
        self.n_plays[i] += 1
        if p2i !=-1:
          self.pop_fits[p2i] += reward2
          self.n_plays[p2i] += 1
        self.env.reset()
    if verbose>0:
      print(self.pop_fits)
    self.pop_fits /= self.n_plays#self.n_plays
    self.pop_fits += fit_laplace
    return self.pop_fits

  def roulette(self, fits=None, verbose=False):
    if fits is None:
      fits = self.pop_fits
      if verbose:
        print("No fitness passed, using stored population fitness")
    if np.min(fits) < 0:
      if verbose:
        print("Fitness negative so moving all fitness up")
      fits -= np.min(fits)
    parent_pool = []
    total_fit = np.sum(fits)
    for i in range(self.pop_size):
      wheel_val = random.random() * total_fit
      p=0
      while wheel_val > fits[p]:
        wheel_val -= fits[p]
        p+=1
      parent_pool.append(self.population[p])
      if verbose:
        print(f"Selected parent {p} with fitness fitness: {fits[p]}")
    self.parent_pool = parent_pool
    return parent_pool
  
  def breed(self, parent1, parent2, verbose=False):
    if verbose:
      print("Breeding children")
    child1 = network(parent1.input_size, parent1.layer_sizes, parent1.act_names)
    child2 = network(parent1.input_size, parent1.layer_sizes, parent1.act_names)
    for i in range(len(parent1.layer_sizes)):
      rows = parent1.weights[i].shape[0]
      cols = parent1.weights[i].shape[1]
      biases = parent1.biases[i].shape[0]
      
      r_cutoff = random.randint(0,parent1.weights[i].shape[0]-1)
      c_cutoff = random.randint(0,parent1.weights[i].shape[1]-1)
      b_cutoff = random.randint(0,parent1.biases[i].shape[0]-1)
      if verbose:
        print(f"Layer [{i}] with shape {parent1.weights[i].shape}\nr_cutoff: {r_cutoff}, c_cutoff: {c_cutoff}\n{r_cutoff*cols+c_cutoff}/{rows*cols}")
      for r in range(rows):
        for c in range(cols):
          if r*cols+c < r_cutoff*cols+c_cutoff:#r<r_cutoff and c<c_cutoff:
            child1.weights[i][r,c] = parent1.weights[i][r,c]
            child2.weights[i][r,c] = parent2.weights[i][r,c]
          else:
            child1.weights[i][r,c] = parent2.weights[i][r,c]
            child2.weights[i][r,c] = parent1.weights[i][r,c]
      for b in range(cols):
        if b<b_cutoff:
          child1.biases[i][b] = parent2.biases[i][b]
          child2.biases[i][b] = parent1.biases[i][b]
        else:
          child1.biases[i][b] = parent1.biases[i][b]
          child2.biases[i][b] = parent2.biases[i][b]
    return child1, child2

  def fast_breed(self, parent1, parent2, verbose=False):
    if verbose:
      print("Breeding children")
    child1 = network(parent1.input_size, parent1.layer_sizes, parent1.act_names)
    child2 = network(parent1.input_size, parent1.layer_sizes, parent1.act_names)
    for i in range(len(parent1.layer_sizes)):
      p1w = parent1.weights[i].flatten()
      p2w = parent2.weights[i].flatten()
      
      cutoff = random.randint(0,p1w.shape[0]-1)
      if verbose:
        print(f"Shape: {parent1.weights[i].shape}, Tot: {p1w.shape}, cutoff: {cutoff}")
        print(f"Concatinated shape: {np.concatenate([p1w[0:cutoff], p2w[cutoff:]]).shape}")
      child1.weights[i] = np.reshape(np.concatenate([p1w[0:cutoff], p2w[cutoff:]]),parent1.weights[i].shape)
      child2.weights[i] = np.reshape(np.concatenate([p2w[0:cutoff], p1w[cutoff:]]),parent1.weights[i].shape)
      
      cutoff = random.randint(0,parent1.biases[i].shape[0]-1)
      child1.biases[i] = np.concatenate([parent1.biases[i][0:cutoff], parent2.biases[i][cutoff:]])
      child2.biases[i] = np.concatenate([parent2.biases[i][0:cutoff], parent1.biases[i][cutoff:]])
    return child1, child2

  def crossover(self, parent_pool=None, verbose=False):
    if parent_pool is None:
      parent_pool = self.parent_pool
    self.child_pool=[]

    for i in range(int(self.pop_size/2)):
      
      p1 = random.randint(0,self.pop_size-1)
      p2 = random.randint(0,self.pop_size-1)
      parent1 = parent_pool[p1]
      parent2 = parent_pool[p2]
      if verbose:
        print(f"\nParent 1: {p1}: ")
        parent1.print_self(verbose=True)
        print("_____________________________________________")
        print(f"\nParent 2: {p2}: ")
        parent2.print_self(verbose=True)
        print("_____________________________________________")

      #child1, child2 = self.fast_breed(parent1, parent2, verbose=verbose)
      child1, child2 = parent1, parent2

      if verbose:
        print(f"\nChild 1: {p1}: ")
        child1.print_self(verbose=True)
        print("_____________________________________________")
        print(f"\nChild 2: {p2}: ")
        child2.print_self(verbose=True)
        print("_____________________________________________")
      self.child_pool.append(child1)
      self.child_pool.append(child2)
  
  def mutate(self, mutation_rate=0.05, epsilon=0.1, verbose=False):
    for i in range(self.pop_size):
      if random.random()<mutation_rate:
        if verbose:
          print(f"Child {i} selected for mutation: ")
          self.child_pool[i].print_self(verbose=True)
        for j in range(len(self.child_pool[i].act_names)):
          shape = self.child_pool[i].weights[j].shape
          self.child_pool[i].weights[j] = np.multiply(self.child_pool[i].weights[j], np.random.rand(shape[0],shape[1])*epsilon -epsilon/2 + 1)
          self.child_pool[i].biases[j] = np.multiply(self.child_pool[i].biases[j], np.random.rand(shape[1])*epsilon -epsilon/2 + 1)
        if verbose:
          print(f"Child {i} after mutation: ")
          self.child_pool[i].print_self(verbose=True)
      elif verbose:
        print(f"Child {i} not chosen")
  def next_gen(self):
    self.population = self.child_pool

