import numpy as np
from tictactoe import env
from network import network
import random
# reinforce code adapted from hugginface RL course

parameters = {
  "pop_size": 100,
  "num_elite": 10,
  "percent_kept": 0.2,
  "min_std": 0.01 
}

net_params ={
  "in_s":9,
  "l_sizes":np.array([64,64,9]),
  "acts": ['relu','relu','sigmoid'],
}

env_params = {
  "win_reward": 1,
  "illegal_reward": 0,
  "legal_reward": 0.1,
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
    
  return r_sum


def cross_entropy():  
  en = env(None,None,None)
  for p in population:
    p.fitness = get_fitness(p,play_rand,10,en,env_params=env_params, verbose=False)

  for p in population:
    print(p.fitness)
  population.sort(key=lambda x: x.fitness, reverse=True)
  for p in population:
    print(p.fitness)
