import numpy as np
from tictactoe import env
from network import network
import random
import matplotlib.pyplot as plt
# reinforce code adapted from hugginface RL course

parameters = {
  "pop_size": 1000,
  "num_elite": 20,
  "percent_kept": 0.2,
  "min_std": 0.002 
}

net_params ={
  "in_s":9,
  "l_sizes":np.array([36,36,9]),
  "acts": ['relu','relu','sigmoid'],
}

env_params = {
  "n_games": 25,
  "win_reward": 1,
  "illegal_reward": 0,
  "legal_reward": -0.01,
  "loss_reward": 0,
}

def play_rand(state):
  move = random.randint(0,8)
  while state[move] != 0:
    move = random.randint(0,8)
  return move

population = []
for i in range(parameters["pop_size"]):
  population.append(network(net_params["in_s"], net_params["l_sizes"], net_params["acts"]))

def get_fitness(network, opponent, n_games, env, env_params, verbose=False):
  r_sum = 0
  players = [None, None]
  for i in range(n_games):
    state = env.reset()[0]
    done, trunc = False, False
    first = random.randint(0,1)
    if verbose:
      print(f"Network will be going {first}")
    players[first] = network.move
    players[1-first] = opponent

    i=0
    while not (done or trunc):
      cur_play = env.current_player
      act = players[cur_play](state)
      state, reward, done, trunc, _ = env.step(act, print_board = verbose, verbose=verbose, legal_reward=env_params["legal_reward"], illegal_reward=env_params["illegal_reward"],win_reward=env_params["win_reward"])
      i+=1
      if cur_play == first:
        r_sum += reward

    # cat get's rewarded too
    if i==9:
      if verbose:
        print(f"Move was cat, adding reward of {env_params['win_reward']/2}")
      r_sum += env_params["win_reward"]/2
    
  return r_sum / n_games


def cross_entropy(population, parameters, env, env_params, net_params, verbose=False):  
  pop_size = parameters["pop_size"]
  for p in population:
    p.fitness = get_fitness(p,play_rand,env_params['n_games'],env,env_params=env_params, verbose=False)
  avg_fit = 0
  population.sort(key=lambda x: x.fitness, reverse=True)
  max_fit = population[0].fitness
  min_fit = population[-1].fitness
  for i in range(pop_size):
    avg_fit += population[i].fitness
  if verbose:
    print("fitnesses after sorting: ", end="")
    for i in range(pop_size):
      if pop_size < 100:
        print(population[i].fitness, end= ", ")
    print(f" avg {avg_fit/pop_size}, max {max_fit}, min {min_fit}")
    
  avg_fit = avg_fit/pop_size
  n1 = population[0]
  weights_mean = []
  biases_mean = []
  # make a place to store the means of weights and biases
  for w in n1.weights:
    weights_mean.append(np.zeros(w.shape))
  for b in n1.biases:
    biases_mean.append(np.zeros(b.shape))

  n_parents = int(parameters["percent_kept"] * parameters["pop_size"])

  for pnum in range(n_parents):
    p = population[pnum]
    for i,w in enumerate(p.weights):
      weights_mean[i] = weights_mean[i] + w
    for i,b in enumerate(p.biases):
      biases_mean[i] = biases_mean[i] + b

  for i in range(len(n1.weights)):
    weights_mean[i] = weights_mean[i] / n_parents
    biases_mean[i] = biases_mean[i] / n_parents
  
  weights_std = []
  biases_std = []
  # make a place to store the means of weights and biases
  for w in n1.weights:
    weights_std.append(np.zeros(w.shape))
  for b in n1.biases:
    biases_std.append(np.zeros(b.shape))
  
  

  for pnum in range(n_parents):
    p = population[pnum]
    for i,w in enumerate(p.weights):
      weights_std[i] = weights_std[i] + np.square(weights_mean[i] - w)
    for i,b in enumerate(p.biases):
      biases_std[i] = biases_std[i] + np.square(biases_mean[i] - b)
  for i in range(len(n1.weights)):
    weights_std[i] = weights_std[i] / pnum + parameters["min_std"] 
    biases_std[i] = biases_std[i] / pnum + parameters["min_std"] 
  
  if verbose and len(population) < 10:
    for p in population:
      p.print_self(verbose=False)
    for i in range(len(weights_mean)):
      print(f"-------------------------w mean: {np.mean(weights_mean[i])}-----------------")
      print(weights_mean[i]) 
      print(f"-------------------------b mean: {np.mean(biases_mean[i])}-----------------")
      print(biases_mean[i]) 
      print(f"-------------------------w std: {np.mean(weights_std[i])}-----------------")
      print(weights_std[i]) 
      print(f"-------------------------b std: {np.mean(biases_std[i])}-----------------")
      print(biases_std[i]) 

  if verbose:
    print(f"-------------------------w mean: {np.mean(np.abs(weights_mean[0]))}-----------------")
    print(f"-------------------------b mean: {np.mean(biases_mean[0])}-----------------")
    print(f"-------------------------w std: {np.mean(weights_std[0])}-----------------")
    print(f"-------------------------b std: {np.mean(biases_std[0])}-----------------")

  new_pop = []
  for i,p in enumerate(population):
    # if elite, pas it on to the next gen
    if i < parameters["num_elite"]:
      new_pop.append(p)
      continue
    
    weights = []
    biases = []

    for i in range(len(weights_mean)):
      weights.append(np.random.normal(weights_mean[i], weights_std[i]))
      biases.append(np.random.normal(biases_mean[i], biases_std[i]))
    new_pop.append(network(net_params["in_s"], net_params["l_sizes"], net_params["acts"], weights, biases))
  return new_pop, avg_fit, max_fit, min_fit

en = env(None, None, None)

pav = []
pmax = []
pmin = []

for i in range(100):
  print(i)
  population, af, mf, minf = cross_entropy(population, parameters, en, env_params, net_params, True)
  pav.append(af)
  pmax.append(mf)
  pmin.append(minf)

  test_state = en.reset()[0]
  complete, truncated = False, False
  while not (complete or truncated):
    act = 0
    cp = en.current_player
    if cp == 0:
      act = population[0].move(test_state)
    else:
      act = play_rand(test_state)
    test_state, rew, complete, truncated, _ = en.step(act, print_board=True, verbose=True, legal_reward=0.1, illegal_reward=-1,win_reward=1)
  


plt.plot(pmin)
plt.plot(pav)
plt.plot(pmax)

plt.legend(["min fit", "ave fit", "max fit"])
plt.show()