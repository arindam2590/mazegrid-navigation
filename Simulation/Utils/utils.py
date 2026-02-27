import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Simulation.envs.maze_env import MazeEnv

def setup_parser():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-n', '--newmaze',
                        action='store_true',
                        help='Start simulation with new maze: use argument -n or --newmaze')
    parser.add_argument('-o', '--oldmaze',
                        action='store_true',
                        help='Start simulation with old maze: use argument -o or --oldmaze')
    parser.add_argument('-d', '--dqn',
                        action='store_true',
                        help='Simulate the environment with DQN Model: use argument -d or --dqn')
    parser.add_argument('-q', '--ddqn',
                        action='store_true',
                        help='Simulate the environment with Double DQN Model: use argument -q or --ddqn')
    parser.add_argument('-u', '--dueldqn',
                        action='store_true',
                        help='Simulate the environment with Dueling DQN Model: use argument -u or --dueldqn')
    parser.add_argument('-p', '--ppo',
                        action='store_true',
                        help='Simulate the environment with PPO Model: use argument -p or --ppo')
    parser.add_argument('-r', '--reinforce',
                        action='store_true',
                        help='Simulate the environment with REINFORCE Model: use argument -r or --reinforce')
    args = parser.parse_args()
    return args


class DataVisualization:
    def __init__(self, episodes, result, model, model_type, trial: int = 1):
        param_dir = 'Simulation/Utils/'
        params = {}
        for cfg in ('env_params.json', 'train_params.json', 'sim_params.json'):
            with open(param_dir + cfg) as f:
                params.update(json.load(f))
        self.params = params

        self.model = model
        self.model_type = model_type
        self.fig_dir = self.params['DATA_DIR'] + self.params['FIG_DIR']
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        self.data_filename = self.params['EXCEL_FILENAME']
        self.n_episodes = episodes
        self.returns = result[0]
        self.epsilon_decay_history = result[1]
        self.training_error = result[2]
        self.steps = result[3]
        # index of all the episodes which got success
        self.success_episodes = result[4]
        self.unique_steps = result[5]
        self.sz   = self.params['SIZE']
        self.epch = trial

        # Load the same maze that the simulation used
        grid_dir        = self.params.get('GRID_MAP_DIR', 'Simulation/grid_map/')
        difficulty      = self.params.get('DIFFICULTY', 'simple')
        self.difficulty = difficulty          # stored for filename
        prefix          = f"{self.sz}x{self.sz}_{difficulty}"
        maze_path  = os.path.join(grid_dir, f'{prefix}_maze.npy')
        loc_path   = os.path.join(grid_dir, f'{prefix}_src_dst.npy')

        self.env = MazeEnv()
        self.env.maze = np.load(maze_path)
        location = np.load(loc_path)
        self.env.source, self.env.destination = location[0], location[1]
        self.env.find_path()
        self.optimal_path_length = len(self.env.path) - 1

    def save_data(self):
        # Filename: e.g.  DQN_200_episode_60x60_simple.xlsx
        # Each trial is a separate sheet:  Trial_1, Trial_2, …
        fname    = (f"{self.model}_{self.n_episodes}_episode_"
                    f"{self.sz}x{self.sz}_{self.difficulty}.xlsx")
        filepath   = self.fig_dir + fname
        sheet_name = f"Trial_{self.epch}"

        if self.model_type == 'VALUE':
            df = pd.DataFrame({'Rewards': self.returns,
                               'Steps': self.steps,
                               'Epsilon Decay': self.epsilon_decay_history,
                               'Training Error': self.training_error,
                               'Success Rate': (len(self.success_episodes)/self.n_episodes)*100,})

            df["Path Efficiency"] = 0.0   # float dtype avoids incompatible-dtype warning
            df["RGEE"] = 0.0
            for ep in range(self.n_episodes):
                # path efficiency: len of optimal path / len of path taken by agent
                df.loc[ep, "Path Efficiency"] = self.optimal_path_length / self.steps[ep]

                # RGEE (reasoning guided exploration): unique states visited / total states visited
                df.loc[ep, "RGEE"] = self.unique_steps[ep] / self.steps[ep]

        else:
            df = pd.DataFrame({'Rewards': self.returns,
                               'Steps': self.steps,
                               'Policy Error': self.training_error[:, 0],
                               'Value Error': self.training_error[:, 1]
                               })

        if not os.path.isfile(filepath):
            # First trial — create workbook fresh
            with pd.ExcelWriter(filepath, mode='w') as writer:
                df.to_excel(writer, sheet_name=sheet_name)
        else:
            # File exists — add / replace only this trial's sheet; keep all others
            with pd.ExcelWriter(filepath, mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=sheet_name)

    def plot_returns(self):
        plot_filename = self.fig_dir + 'MazeGrid_' + self.model + '_cumm_training_returns.png'

        sum_rewards = np.zeros(self.n_episodes)
        for episode in range(self.n_episodes):
            sum_rewards[episode] = np.sum(self.returns[0:(episode + 1)])

        # Plot and save the cumulative return graph
        plt.plot(sum_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Returns')
        plt.title('Cumulative Reward per Episodes')
        plt.savefig(plot_filename)
        plt.clf()

        plot_filename = self.fig_dir + 'MazeGrid_' + self.model + '_training_returns.png'
        plt.plot(self.returns)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('Returns per Episodes')
        plt.savefig(plot_filename)
        plt.clf()

    def plot_episode_length(self):
        plot_filename = self.fig_dir + 'MazeGrid_' + self.model + '_training_episode_len.png'
        plt.plot(range(self.n_episodes), self.steps)
        plt.xlabel('Episodes')
        plt.ylabel('Episode Length')
        plt.title('Episode Length per Episode')
        plt.savefig(plot_filename)
        plt.clf()

    def plot_epsilon_decay(self):
        plot_filename = self.fig_dir + 'MazeGrid_' + self.model + '_epsilon_decay.png'
        plt.plot(range(self.n_episodes), self.epsilon_decay_history)
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon Decay')
        plt.title('Epsilon Decay per Episode')
        plt.savefig(plot_filename)
        plt.clf()

    def plot_training_error(self,):
        if self.model_type == 'VALUE':
            plot_filename = self.fig_dir + 'MazeGrid_' + self.model + '_training_error.png'
            plt.plot(range(self.n_episodes), self.training_error)
            plt.xlabel('Episodes')
            plt.ylabel('Temporal Difference')
            plt.title('Temporal Difference per Episode')
            plt.savefig(plot_filename)
            plt.clf()
        else:
            plot_filename = self.fig_dir + 'MazeGrid_' + self.model + '_training_error.png'
            plt.plot(range(self.n_episodes), self.training_error[:, 0], label='policy loss')
            plt.plot(range(self.n_episodes), self.training_error[:, 1], label='value loss')
            plt.xlabel('Episodes')
            plt.ylabel('Model Loss')
            plt.title('Policy and Value Loss per Episode')
            plt.legend()
            plt.savefig(plot_filename)
            plt.clf()
