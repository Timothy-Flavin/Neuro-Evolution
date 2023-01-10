import numpy as np

# The board is 18 bits, the 9 positions for x and the 9 positions for o
# x goes first. 

class env:
  def __init__(self, player1, player2, board=None):
    self.player1 = player1
    self.player2 = player2
    if board is not None:
      self.board = board
    else:
      self.board = np.zeros(18)
  
  def reset(self):
    self.board = np.zeros(18)

  def check_win(self, board=None):
    if board is None:
      board = self.board

    indices = np.array([
      [0,1,2],
      [3,4,5],
      [6,7,8],
      [0,3,6],
      [1,4,7],
      [2,5,8],
      [0,4,8],
      [2,4,6],
    ])
    for i in range(indices.shape[0]):
      if np.array_equal(board[indices[i,:]],np.ones(3)):
        return True
    for i in range(indices.shape[0]):
      if np.array_equal(board[indices[i,:]+9],np.ones(3)):
        return True

    else:
      return False

  def print_board(self, board=None):
    if board is None:
      board = self.board
    
    for r in range(3):
      for c in range(3):
        if board[c+3*r] == 1:
          print('x',end=' ')
        elif board[c+3*r+9] == 1:
          print('o',end=' ')
        else:
          print('-',end=' ')
      print()

# returns 1 if player 1 wins and 0.5 if draw and 0 if loss  
  def play(self, players=None, print_board = False, verbose=False):
    if players is not None:
      self.player1 = players[0]
      self.player2 = players[1]
    
    if print_board:
      self.print_board()
    
    nmoves = 0
    while nmoves < 10:
      if verbose:
        print("Player 1 moves")
      move = self.player1.move(self.board)
      # if the space chosen is free
      if self.board[move] == 0 and self.board[move+9] == 0:
        self.board[move] = 1
        nmoves +=1
      else:
        if verbose:
          print("Player 1 loses from illegal move")
        return 0
      if print_board:
        #print(self.board)
        self.print_board()
      won = self.check_win()
      if won:
        if verbose:
          print("Player 1 wins!")
        return 1
      if nmoves == 9:
        if verbose:
          print("Draw")
        return 0.5
    # Player 2's turn, player 2 sees a board where x and y are flipped
      if verbose:
        print("Player 2 moves")
      board2 = np.zeros(self.board.shape[0])
      board2[0:9] = self.board[9:18]
      board2[9:18] = self.board[0:9]

      move = self.player1.move(board2)
      # if the space chosen is free
      if self.board[move] == 0 and self.board[move+9] == 0:
        self.board[move+9] = 1
        nmoves +=1
      else:
        if verbose:
          print("Player 2 loses from illegal move")
        return 1
      if print_board:
        self.print_board()
      won = self.check_win()
      if won:
        if verbose:
          print("Player 2 wins!")
        return 0
      