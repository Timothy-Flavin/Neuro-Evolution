import numpy as np
from neuroevolution import NeuralGA
from tictactoe import env

ga = NeuralGA(4, env=env(None,None,None), input_size=18,layer_sizes=np.array([3,9]),activations=["relu","sigmoid"])

fits = ga.get_fitness_ttt(num_plays=5,epsilon=0.35,fit_laplace=0.01,verbose=1)
print(ga.n_plays)
print(fits)
ga.roulette(verbose=True)
print("------------------------------------------------")
ga.crossover(verbose=True)
ga.mutate(mutation_rate=0.5,verbose=True)
ga.next_gen()


print("Silent")
fits = ga.get_fitness_ttt(num_plays=3,epsilon=0.5,fit_laplace=0.05,verbose=0)
ga.roulette(verbose=False)
ga.crossover(verbose=False)
ga.mutate(mutation_rate=0.5,verbose=False)
ga.next_gen()
print("Done")