import numpy as np
from network import network

mynet = network(3, np.array([2,1]),activations=['relu','sigmoid'])

mynet.print_self(verbose=True)

print("--------------------------------------")

mynet.feed_forward(np.array([1,2,-1]),verbose=True)