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

    self.current_player = True

  def reset(self, bsize=9):
    
    self.board = np.zeros(bsize)
    self.current_player = 0
    self.nmoves = 0  
    return [self.board]

  def check_win_old(self, board=None):
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
      if np.array_equal(board[indices[i,:]],-1*np.ones(3)):
        return True
    else:
      return False

  def print_board_old(self, board=None):
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

  def print_board(self, board=None):
    if board is None:
      board = self.board
    
    for r in range(3):
      for c in range(3):
        if board[c+3*r] == 1:
          print('x',end=' ')
        elif board[c+3*r] == -1:
          print('o',end=' ')
        else:
          print('-',end=' ')
      print()
# returns 1 if player 1 wins and 0.5 if draw and 0 if loss  
  def play_simple(self, players=None, print_board = False, verbose=False):
    print("hi")
    if players is not None:
      self.player1 = players[0]
      self.player2 = players[1]
    
    if print_board:
      self.print_board()
    
    nmoves = 0
    while nmoves < 10:
      move = self.player1.move(self.board, verbose=verbose)
      if verbose:
        print(f"Player 1 moves: {move}")
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
      
      board2 = np.zeros(self.board.shape[0])
      board2[0:9] = self.board[9:18]
      board2[9:18] = self.board[0:9]

      move = self.player2.move(board2, verbose=verbose)
      if verbose:
        print(f"Player 2 moves {move}")
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
      
  def play(self, players=None, print_board = False, verbose=False, legal_reward=0.1, illegal_reward=-1,win_reward=1):
    p1reward = 0
    p2reward = 0
    if players is not None:
      self.player1 = players[0]
      self.player2 = players[1]
    
    if print_board:
      self.print_board()
    
    nmoves = 0
    while nmoves < 10:
      move = self.player1.move(self.board, verbose=verbose)
      if verbose:
        print(f"Player 1 moves: {move}")
      # if the space chosen is free
      if self.board[move] == 0 and self.board[move+9] == 0:
        self.board[move] = 1
        nmoves +=1
        p1reward += legal_reward
      else:
        if verbose:
          print("Player 1 loses from illegal move")
        p1reward += illegal_reward
        return p1reward, p2reward
      if print_board:
        #print(self.board)
        self.print_board()
      won = self.check_win()
      if won:
        if verbose:
          print("Player 1 wins!")
        p1reward += win_reward
        return p1reward, p2reward
      if nmoves == 9:
        if verbose:
          print("Draw")
        return p1reward, p2reward
    # Player 2's turn, player 2 sees a board where x and y are flipped
      
      board2 = np.zeros(self.board.shape[0])
      board2[0:9] = self.board[9:18]
      board2[9:18] = self.board[0:9]

      move = self.player2.move(board2, verbose=verbose)
      if verbose:
        print(f"Player 2 moves {move}")
      # if the space chosen is free
      if self.board[move] == 0 and self.board[move+9] == 0:
        self.board[move+9] = 1
        p2reward+=legal_reward
        nmoves +=1
      else:
        if verbose:
          print("Player 2 loses from illegal move")
        p2reward+=illegal_reward
        return p1reward,p2reward
      if print_board:
        self.print_board()
      won = self.check_win()
      if won:
        if verbose:
          print("Player 2 wins!")
        p2reward+=win_reward
        return p1reward,p2reward

  def step(self, action, print_board = False, verbose=False, legal_reward=0.1, illegal_reward=-1,win_reward=1):
    reward = 0    
    if self.nmoves == 0:
      self.current_player = 0
    
    board = self.board

    if verbose:
      print(f"Player {self.current_player} moves: {action}")
    # if the space chosen is free
    if self.board[action] == 0:
      if self.current_player == 0:
        self.board[action] = 1
      else:
        self.board[action] = -1
      self.nmoves += 1
      if print_board:
        self.print_board()
      reward += legal_reward
      won = self.check_win()

      if won:
        #print(f"Player {self.current_player} wins!")
        if verbose:
          print(f"Player {self.current_player} wins!")
        reward += win_reward
        #observation, reward, terminated, truncated, info
        self.current_player = 1-self.current_player
        return board, reward, True, False, None
      if self.nmoves == 9:
        if verbose:
          print("Draw")
        self.current_player = 1-self.current_player
        return board, reward, False, True, None
      self.current_player = 1-self.current_player
      return board, reward, False, False, None
    else:
      if verbose:
        print(f"Player {self.current_player} loses from illegal move")
      reward += illegal_reward
      self.current_player = 1-self.current_player
      return board, reward, False, True, None


def step_old(self, action, print_board = False, verbose=False, legal_reward=0.1, illegal_reward=-1,win_reward=1):
    reward = 0    
    if self.nmoves == 0:
      self.current_player = 0
    
    board = self.board

    if verbose:
      print(f"Player {self.current_player} moves: {action}")
    # if the space chosen is free
    if self.board[action] == 0 and self.board[action+9] == 0:
      self.board[action + self.current_player*9] = 1
      self.nmoves += 1
      if print_board:
        self.print_board()
      reward += legal_reward
      won = self.check_win()

      # if player 0 is playing, player 2 will need to see
      # the flipped board
      if self.current_player == 0:
        board = np.zeros(self.board.shape[0])
        board[0:9] = self.board[9:18]
        board[9:18] = self.board[0:9]

      if won:
        #print(f"Player {self.current_player} wins!")
        if verbose:
          print(f"Player {self.current_player} wins!")
        reward += win_reward
        #observation, reward, terminated, truncated, info
        self.current_player = 1-self.current_player
        return board, reward, True, False, None
      if self.nmoves == 9:
        if verbose:
          print("Draw")
        self.current_player = 1-self.current_player
        return board, reward, False, True, None
      self.current_player = 1-self.current_player
      return board, reward, False, False, None
    else:
      if verbose:
        print(f"Player {self.current_player} loses from illegal move")
      reward += illegal_reward
      self.current_player = 1-self.current_player
      return board, reward, False, True, None

