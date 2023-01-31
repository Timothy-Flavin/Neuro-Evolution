import numpy as np
from tictactoe import env

en = env(None, None)

en.reset()
done = False
truncated = False
en.print_board()

while not done and not truncated:
  action = int(input("Input action: "))
  state, reward, done, truncated, info = en.step(action, True, True)
  print(f"reward: {reward}, state {state}")