import numpy as np
from network import network
import random

class NeuralGA:
  
  def default_fitness(self, game, players):
    return game.play(players, print_board = False, verbose = False)

  def __init__(self, pop_size, env, fit_func = default_fitness, input_size=18, layer_sizes=np.array([32, 16, 9]), activations=["relu","relu","sigmoid"]):
    self.pop_size = pop_size
    self.population = []
    self.pop_fits = np.zeros(pop_size)
    for i in range(pop_size):
      self.population.append(network(input_size, layer_sizes, activations))
    self.fit_func = fit_func
    self.env = env
  
  def get_fitness_ttt(self, num_plays):
    # for each player, get their fittness
    for i in range(self.pop_size):
      self.pop_fits[i] = 0
      # play num_plays games and get average return
      for pnum in range(num_plays):
        self.pop_fits += self.env.play(players=[self.population[i], self.population[random.randint(0,self.popsize-1)]], print_board = False, verbose=False)
      self.pop_fits[i] /= num_plays
    return self.pop_fits

  def roulette(self, fits=None):
    if fits is None:
      fits = self.pop_fits
    parent_pool = []
    total_fit = np.sum(fits)
    for i in range(self.pop_size):
      wheel_val = random.random() * total_fit
      p=0
      while wheel_val > self.pop_fits[p]:
        wheel_val -= self.pop_fits[p]
        p+=1
      parent_pool.append(self.population[p])
    self.parent_pool = parent_pool
    return parent_pool
  
  def breed(self, parent1, parent2):
    print("breedin")
    
  def crossover(self, parent_pool=None):
    if parent_pool is None:
      parent_pool = self.parent_pool
    for i in range(self.pop_size/2):
      parent1 = parent_pool[random.randint(0,self.pop_size-1)]
      parent2 = parent_pool[random.randint(0,self.pop_size-1)]

