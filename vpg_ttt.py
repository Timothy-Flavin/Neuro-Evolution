import torch
import torch.optim as optim
import numpy as np
from tictactoe import env
from torch_network import LeakyMLP, SigMLP
from collections import deque
import random
# reinforce code adapted from hugginface RL course

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def play_rand(state):
  move = random.randint(0,8)
  while state[move] != 0:
    move = random.randint(0,8)
  return move
#policy1, policy2, optimizer1, optimizer2,
def reinforce(policy1, policy2, optimizer1, optimizer2, n_training_episodes, gamma, print_every, env, legal_reward=0.1, illegal_reward=-1,win_reward=1, loss_reward=-1, epsilon=0.8):
  # Help us to calculate the score during the training
  scores1 = []
  scores2 = []
  # Line 3 of pseudocode
  for i_episode in range(1, n_training_episodes + 1):
    epsilon = 0.99999 * epsilon
    saved_log_probs_p1 = []
    rewards_p1 = []
    saved_log_probs_p2 = []
    rewards_p2 = []

    p1_rand = False
    p2_rand = False
    if random.random() < epsilon:
      p1_rand = True
    elif random.random() < epsilon:
      p2_rand = True

    state = env.reset()[0]
    # Line 4 of pseudocode tic tac toe can only go 10 moves
    for t in range(10):
      #print(state)
      action, log_prob = 0, 0
      cur_play = env.current_player
      if cur_play == 0:
        if p1_rand:
          action = play_rand(state)
        else:
          action, log_prob = policy1.act(state)
      else: 
        if p2_rand:
          action = play_rand(state)
        else:
          action, log_prob = policy2.act(state)
      state, reward, done, doner, _ = env.step(action, print_board=False, verbose=False, legal_reward=legal_reward, illegal_reward=illegal_reward,win_reward=win_reward)
      #input()
      #print(reward)
      if cur_play == 0:
        if not p1_rand:
          saved_log_probs_p1.append(log_prob)
          rewards_p1.append(reward)
      else:
        if not p2_rand:
          saved_log_probs_p2.append(log_prob)
          rewards_p2.append(reward)
      
      if done or doner:
        if done: # done is terminated, doner is from illegal moves or cat
          if cur_play == 0:
            #print("player 2 lost")
            if not p2_rand:
              rewards_p2[-1] += loss_reward
          else:
            #print("player 1 lost")
            if not p1_rand:
              rewards_p1[-1] += loss_reward
        break
    
    if not p1_rand:
      scores1.append(sum(rewards_p1))
    if not p2_rand:
      scores2.append(sum(rewards_p2))
    #print(f"r1: {sum(rewards_p1)}, {sum(rewards_p2)}")

    # Line 6 of pseudocode: calculate the return
    returns1, returns2, n_steps1, n_steps2 = [],[],0,0
    if not p1_rand:
      returns1 = deque(maxlen=10)
      n_steps1 = len(rewards_p1)

    if not p2_rand:
      returns2 = deque(maxlen=10)
      n_steps2 = len(rewards_p2)

    if not p1_rand:
      # calculating performance as player 1
      for t in range(n_steps1)[::-1]:
        disc_return_t = returns1[0] if len(returns1) > 0 else 0
        returns1.appendleft(gamma * disc_return_t + rewards_p1[t])
      returns1 = torch.tensor(returns1)
    if not p2_rand:
      # calculating performance as player 2
      for t in range(n_steps2)[::-1]:
        disc_return_t = returns2[0] if len(returns2) > 0 else 0
        returns2.appendleft(gamma * disc_return_t + rewards_p2[t])
      returns2 = torch.tensor(returns2)
    ## standardization of the returns is employed to make training more stable
    eps = np.finfo(np.float32).eps.item()
    ## eps is the smallest representable float, which is
    # added to the standard deviation of the returns to avoid numerical instabilities

    #if len(returns1)<0:
      #returns1 = (returns1 - returns1.mean()) / (returns1.std() + eps)
    #if len(returns2)<0:
      #returns2 = (returns2 - returns2.mean()) / (returns2.std() + eps)
 
    # Line 7:
    policy_loss1 = []
    policy_loss2 = []
    if not p1_rand:
      for log_prob, disc_return in zip(saved_log_probs_p1, returns1):
        policy_loss1.append(-log_prob * disc_return)
      policy_loss1 = torch.cat(policy_loss1).sum()
    if not p2_rand:
      for log_prob, disc_return in zip(saved_log_probs_p2, returns2):
        policy_loss2.append(-log_prob * disc_return)
      policy_loss2 = torch.cat(policy_loss2).sum()

    # Line 8: PyTorch prefers gradient descent
    if not p1_rand:
      optimizer1.zero_grad()
      policy_loss1.backward()
      optimizer1.step()
    #print("hi")
    #input("made it past step 1")
    if not p2_rand:
      optimizer2.zero_grad()
      policy_loss2.backward()
      optimizer2.step()

    if i_episode % print_every == 0:
      print("Episode {}\tAverage Score1: {:.2f} Score2: {:.2f}".format(i_episode, np.mean(scores1), np.mean(scores2)))
      print(f"Epsilon: {epsilon}")
      test_state = env.reset()[0]
      complete, truncated = False, False
      while not (complete or truncated):
        act = 0
        cp = env.current_player
        if cp == 0:
          act, lp = policy1.act(test_state)
        else:
          act, lp = policy2.act(test_state)
        test_state, rew, complete, truncated, _ = env.step(act, print_board=True, verbose=True, legal_reward=legal_reward, illegal_reward=illegal_reward,win_reward=win_reward)
        
  
  return scores1, scores2, policy1, policy2




ttt_hyperparameters = {
    "h_sizes": [128,64,18],
    "n_training_episodes": 1000000,
    "n_evaluation_episodes": 10,
    "gamma": 1.0,
    "lr": 5e-3,
    "env_id": "TicTacToe",
    "state_space": 9,
    "action_space": 9,
    "win_reward": 1,
    "illegal_punish": 0,
    "legal_reward": 0.5,
    "loss_reward": 0,
}

ttt_policy1 = LeakyMLP(
    ttt_hyperparameters["state_space"],
    ttt_hyperparameters["action_space"],
    ttt_hyperparameters["h_sizes"],
).to(device)
ttt_policy2 = LeakyMLP(
    ttt_hyperparameters["state_space"],
    ttt_hyperparameters["action_space"],
    ttt_hyperparameters["h_sizes"],
).to(device)

ttt_optimizer1 = optim.Adam(ttt_policy1.parameters(), lr=ttt_hyperparameters["lr"])
ttt_optimizer2 = optim.Adam(ttt_policy2.parameters(), lr=ttt_hyperparameters["lr"])

#setting up environment
en = env(None, None)
en.reset()
#torch.autograd.set_detect_anomaly(True)
scores1, scores2, policy1, policy2 = reinforce(
    ttt_policy1,
    ttt_policy2,
    ttt_optimizer1,
    ttt_optimizer2,
    ttt_hyperparameters["n_training_episodes"],
    ttt_hyperparameters["gamma"],
    1000,
    en,
    legal_reward=ttt_hyperparameters["legal_reward"],
    illegal_reward=ttt_hyperparameters["illegal_punish"],
    win_reward=ttt_hyperparameters["win_reward"],
    loss_reward=ttt_hyperparameters["loss_reward"],
)

state = en.reset()[0]
done = False
truncated = False
en.print_board()

while True:
  human = random.randint(0,1)
  action=0
  log_prob = 0
  done=False
  truncated=False
  while not (done or truncated):
    print(f"current player: {en.current_player}")
    if en.current_player == human:
      action = int(input("Input action: "))
    else:
      if en.current_player == 0:
        print(f"policy1 played: {action} with probability {log_prob}")
        action, log_prob = policy1.act(state)
      else:
        print(f"policy2 played: {action} with probability {log_prob}")
        action, log_prob = policy2.act(state)
    state, reward, done, truncated, info = en.step(action, True, True)
    print(f"reward: {reward}, state {state}, done: {done}, truncated: {truncated}")
    #input("next turn?")
  state = en.reset()[0]