import numpy as np
from tictactoe import env

class human:
  def move(self, board):
    return int(input("Select move 0-8:"))

en = env([human(), human()], np.zeros(18))
en.check_win()
for i in range(100):
  b = np.random.randint(0,2,size=18)
  w = en.check_win(b)
  print("______________________________________")
  print(w)
  print(b)
  en.print_board(board = b)
  print("______________________________________")

en.play(print_board=True, verbose=True)
en.reset()
print("Game 2")
en.play(print_board=True, verbose=True)