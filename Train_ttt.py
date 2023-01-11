from network import network

class human:
  def move(self, board):
    return int(input("Select move 0-8:"))

import numpy as np
from neuroevolution import NeuralGA
from tictactoe import env

ga = NeuralGA(500, env=env(None,None,None), input_size=18,layer_sizes=np.array([32,18,9]),activations=["relu","relu","relu"])

num_gens = 1000
eps = 0.1
avg_fits = []
best_fits = []
fits=None
for i in range(num_gens):
  eps = i/num_gens/2
  fits = ga.get_fitness_ttt(num_plays=10,epsilon=eps,fit_laplace=0.05,verbose=0)
  if i%10==0:
    print(f"Progress: {100*i/num_gens}%, avg fit: {np.mean(fits)}, best: {np.max(fits)}")
    avg_fits.append(np.mean(fits))
    best_fits.append(np.max(fits))
  if i%1000 == 0:
    
    print(fits)
    #pl = input("Want to play? y or n")
    pl = 'n'
    if pl=='y':
      print(f"fittest player: {np.argmax(fits)}")
      best_player = ga.population[np.argmax(fits)]
      print("Game 1: ")
      en = env(best_player, human(), np.zeros(18))
      en.play(print_board=True, verbose=True)
      en.reset()
      print("Game 2: ")
      en = env(human(),best_player, np.zeros(18))
      en.play(print_board=True, verbose=True)
      en.reset()
  ga.roulette()
  ga.crossover()
  ga.mutate(mutation_rate=0.1,verbose=False)
  ga.next_gen()


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