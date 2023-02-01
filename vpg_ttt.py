import torch
import torch.optim as optim
import numpy as np
from tictactoe import env
from torch_network import LeakyMLP
from collections import deque
import random
# reinforce code adapted from hugginface RL course

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#policy1, policy2, optimizer1, optimizer2,
def reinforce(policy, optimizer, n_training_episodes, gamma, print_every, env, legal_reward=0.1, illegal_reward=-1,win_reward=1, loss_reward=-1):
  # Help us to calculate the score during the training
  scores = []
  # Line 3 of pseudocode
  for i_episode in range(1, n_training_episodes + 1):
    saved_log_probs_p1 = []
    rewards_p1 = []
    saved_log_probs_p2 = []
    rewards_p2 = []

    state = env.reset()[0]
    # Line 4 of pseudocode tic tac toe can only go 10 moves
    for t in range(10):
      #print(state)
      action, log_prob = 0, 0
      if env.current_player == 0:
        action, log_prob = policy.act(state)
      else: 
        action, log_prob = policy.act(state)
      state, reward, done, doner, _ = env.step(action, print_board=False, verbose=False, legal_reward=legal_reward, illegal_reward=illegal_reward,win_reward=win_reward)
      #input()
      #print(reward)
      if env.current_player == 0:
        saved_log_probs_p1.append(log_prob)
        rewards_p1.append(reward)
      else:
        saved_log_probs_p2.append(log_prob)
        rewards_p2.append(reward)
      
      if done or doner:
        if done: # done is terminated, doner is from illegal moves or cat
          if env.current_player == 0:
            #print("player 2 lost")
            rewards_p2[-1] += loss_reward
          else:
            #print("player 1 lost")
            rewards_p1[-1] += loss_reward
        break

    scores.append(sum(rewards_p1))
    scores.append(sum(rewards_p2))
    #print(f"r1: {sum(rewards_p1)}, {sum(rewards_p2)}")

    # Line 6 of pseudocode: calculate the return
    returns1 = deque(maxlen=10)
    returns2 = deque(maxlen=10)
    n_steps1 = len(rewards_p1)
    n_steps2 = len(rewards_p2)

    # calculating performance as player 1
    for t in range(n_steps1)[::-1]:
      disc_return_t = returns1[0] if len(returns1) > 0 else 0
      returns1.appendleft(gamma * disc_return_t + rewards_p1[t])
    # calculating performance as player 2
    for t in range(n_steps2)[::-1]:
      disc_return_t = returns2[0] if len(returns2) > 0 else 0
      returns2.appendleft(gamma * disc_return_t + rewards_p2[t])

    ## standardization of the returns is employed to make training more stable
    eps = np.finfo(np.float32).eps.item()
    ## eps is the smallest representable float, which is
    # added to the standard deviation of the returns to avoid numerical instabilities
    returns1 = torch.tensor(returns1)
    returns2 = torch.tensor(returns2)
    #print("returns 1 and 2 before norm")
    #print(returns1.std())
    #print(returns2.std())
    if len(returns1)<0:
      returns1 = (returns1 - returns1.mean()) / (returns1.std() + eps)
    if len(returns2)<0:
      returns2 = (returns2 - returns2.mean()) / (returns2.std() + eps)
    
    #print("returns 1 and 2 after norm")
    #print(returns1)
    #print(returns2)
    #print("rewards 1 and 2")
    #print(rewards_p1)
    #print(rewards_p2)
    # Line 7:
    policy_loss = []
    for log_prob, disc_return in zip(saved_log_probs_p1, returns1):
      policy_loss.append(-log_prob * disc_return)
    for log_prob, disc_return in zip(saved_log_probs_p2, returns2):
      policy_loss.append(-log_prob * disc_return)
    
    #print(policy_loss)
    policy_loss = torch.cat(policy_loss).sum()
    #print(policy_loss)
    #input("loss look ok?")
    # Line 8: PyTorch prefers gradient descent
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    if i_episode % print_every == 0:
        print("Episode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores)))

  return scores, policy




ttt_hyperparameters = {
    "h_sizes": [128,64,32,12],
    "n_training_episodes": 10000,
    "n_evaluation_episodes": 10,
    "gamma": 1.0,
    "lr": 1e-2,
    "env_id": "TicTacToe",
    "state_space": 18,
    "action_space": 9,
    "win_reward": 2,
    "illegal_punish": -1,
    "legal_reward": 0,
    "loss_reward": -0.5,
}

ttt_policy = LeakyMLP(
    ttt_hyperparameters["state_space"],
    ttt_hyperparameters["action_space"],
    ttt_hyperparameters["h_sizes"],
).to(device)
ttt_optimizer = optim.Adam(ttt_policy.parameters(), lr=ttt_hyperparameters["lr"])

#setting up environment
en = env(None, None)
en.reset()

scores, policy = reinforce(
    ttt_policy,
    ttt_optimizer,
    ttt_hyperparameters["n_training_episodes"],
    ttt_hyperparameters["gamma"],
    500,
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
      #action = int(input("Input action: "))
      print(f"policy played: {action} with probability {log_prob}")
      action, log_prob = policy.act(state)
    else:
      print(f"policy played: {action} with probability {log_prob}")
      action, log_prob = policy.act(state)
    state, reward, done, truncated, info = en.step(action, True, True)
    print(f"reward: {reward}, state {state}, done: {done}, truncated: {truncated}")
    input("next turn?")
  state = en.reset()[0]