from network import network
import time
class human:
  def move(self, board, verbose=False):
    return int(input("Select move 0-8:"))

import numpy as np
from neuroevolution import NeuralGA
from tictactoe import env

ga = NeuralGA(200, env=env(None,None,None), input_size=18,layer_sizes=np.array([24,18,14,12,9]),activations=["relu","relu","relu","relu","sigmoid"], legal_reward=-0.1, illegal_reward=-2,win_reward=2)

num_gens = 1000
eps = 0.1
avg_fits = []
best_fits = []
fits=None
start_time = time.time()

fit_time = 0
roulette_time = 0
crossover_time = 0
mutate_time = 0

for i in range(num_gens):
  eps = i/num_gens/2 + 0.5
  temp_time = time.time()
  fits = ga.get_fitness_ttt(num_plays=5,epsilon=eps,fit_laplace=0.01,verbose=0)
  fit_time += time.time()-temp_time
  if i%20==0:
    print(f"Progress: {100*i/num_gens}%, avg fit: {np.mean(fits)}, best: {np.max(fits)}")
    avg_fits.append(np.mean(fits))
    best_fits.append(np.max(fits))
  if i%200 == 0 and i>0:
    
    print(fits)
    pl = input("Want to play? y or n")
    #pl = 'n'
    if pl=='y':
      print(f"fittest player: {np.argmax(fits)}")
      best_player = ga.population[np.argmax(fits)]
      print("Game 1: ")
      en = env(best_player, human(), np.zeros(18))
      en.play(print_board=True, verbose=True)
      en.reset()
      print("Game 2: ")
      best_player.set_move_type(mtype = "max_legal")
      en = env(best_player, human(), np.zeros(18))
      en.play(print_board=True, verbose=True)
      en.reset()
      print("Game 3: ")
      en = env(human(),best_player, np.zeros(18))
      en.play(print_board=True, verbose=True)
      en.reset()
      print("Game 4: ")
      best_player.set_move_type(mtype = "prob_legal")
      en = env(human(),best_player, np.zeros(18))
      en.play(print_board=True, verbose=True)
      en.reset()
  temp_time = time.time()
  ga.roulette()
  roulette_time += time.time() - temp_time
  temp_time = time.time()
  ga.crossover()
  crossover_time += time.time() - temp_time
  temp_time = time.time()
  ga.mutate(mutation_rate=0.1,verbose=False)
  mutate_time += time.time() - temp_time
  ga.next_gen()

  if i%20==0:
    print(f"Time Elapsed: {time.time() - start_time}. fit: {fit_time}, roulette: {roulette_time}, cross: {crossover_time}, mut: {mutate_time}")

print(avg_fits)
print(best_fits)
print(f"fittest player: {np.argmax(fits)}")
best_player = ga.population[np.argmax(fits)]
print("Game 1: ")
en = env(best_player, human(), np.zeros(18))
en.play(print_board=True, verbose=True)
en.reset()
print("Game 2: ")
en = env(best_player, human(), np.zeros(18))
en.play(print_board=True, verbose=True)
en.reset()
print("Game 3: ")
en = env(human(),best_player, np.zeros(18))
en.play(print_board=True, verbose=True)
en.reset()
print("Game 4: ")
en = env(human(),best_player, np.zeros(18))
en.play(print_board=True, verbose=True)
en.reset()