import numpy as np
from ple import PLE
from ple.games.catcher import Catcher


class PlayingAgent:

    def __init__(self, actions, model):
        self.actions = actions
        self.theta = model

    def pick_action(self, obs):
        obs = list(game.getGameState().values())
        obs = np.hstack((1, obs))

        if np.dot(self.theta, obs) > 0:
            action = 97
        else:
            action = 100
        return action


# load trained model and descrete states
model = np.load('../model/cem_theta.npy')

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