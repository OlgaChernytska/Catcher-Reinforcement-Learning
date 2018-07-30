import numpy as np
from ple import PLE
from ple.games.catcher import Catcher
import torch
import torch.nn.functional as F
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.fc2(F.sigmoid(self.fc1(x)))
        return x

class PlayingAgent:
    def __init__(self, actions, model):
        self.actions = actions
        self.model = model

    def pick_action(self, obs):
        obs = np.resize(np.array(list(obs.values())), (1, 4))
        # normalize state characteristics
        norm_coeff = np.resize([100, 10, 100, 100], (1, 4))
        obs = obs / norm_coeff

        arr = np.hstack((np.identity(3), np.tile(obs, (3, 1))))
        inputs = torch.FloatTensor(arr)
        outputs = self.model(inputs)
        _, action_index = outputs.max(0)
        action_index = int(action_index)
        action = self.actions[action_index]
        return action

# load trained neural network
model = torch.load('../model/neural_network.pt')

# initialize game
game = Catcher(width=100, height=100, init_lives=1)
p = PLE(game, fps=30, frame_skip=3, num_steps=1, force_fps=False, display_screen=True)
p.init()

# initialize agent
agent = PlayingAgent(p.getActionSet(), model)

# run training
episodes = 10
max_timestamps = 300

episode_results = []

for episode_index in range(episodes):
    p.reset_game()

    for timestamp in range(max_timestamps):
        observation = game.getGameState()
        action = agent.pick_action(observation)
        reward = p.act(action)

        if p.game_over():
            break

    print('Episode %5d Timestamps: %d' % (episode_index + 1, timestamp + 1))