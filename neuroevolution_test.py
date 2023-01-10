import numpy as np
from neuroevolution import NeuralGA
from tictactoe import env

ga = NeuralGA(4, env=env(None,None,None), input_size=18,layer_sizes=np.array([3,9]),activations=["relu","sigmoid"])

fits = ga.get_fitness_ttt(3,1)
print(fits)
ga.roulette(verbose=True)
print("------------------------------------------------")
ga.crossover(verbose=True)
ga.mutate(mutation_rate=0.5,verbose=True)
ga.next_gen()


print("Silent")
fits = ga.get_fitness_ttt(3,0)
ga.roulette(verbose=False)
ga.crossover(verbose=False)
ga.mutate(mutation_rate=0.5,verbose=False)
ga.next_gen()
print("Done")