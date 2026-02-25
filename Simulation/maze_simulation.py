import json
import os
import time
import pygame
import numpy as np
from .agent import Agent
from .envs.maze_env import MazeEnv
from .model.dqntorch import DQNModel,DoubleDQNModel,DuelingDQNModel
from .Utils.utils import DataVisualization


class Simulation:
    def __init__(self, args, train_mode, train_episodes=100, render=False):
        param_dir = 'Simulation/Utils/'

        # Load separate parameter files
        with open(param_dir + 'env_params.json', 'r') as f:
            self.env_params = json.load(f)
        with open(param_dir + 'train_params.json', 'r') as f:
            self.train_params = json.load(f)
        with open(param_dir + 'sim_params.json', 'r') as f:
            self.sim_params = json.load(f)

        # Merge all params for convenience (existing code uses self.params[...])
        self.params = {**self.env_params, **self.train_params, **self.sim_params}

        print(f'\n' + '%' * 50 + ' Deep Reinforcement Learning Based MAZE Solver ' + '%' * 50)
        self.data_dir = self.params['DATA_DIR'] + self.params['MAZE_DIR']
        filename = self.params['MAZE_FILENAME']

        if args.newmaze:
            self.is_new_map = True
            print(f'Info: Simulation has been started with New Map')
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
        else:
            self.is_new_map = False
            print(f'Info: Simulation has been started with Old Map')
            if not os.path.exists(self.data_dir):
                print(f'Exception: Data directory does not exist. Unable to load saved maze!!')
                exit(0)
            else:
                file_path = self.data_dir + filename
                if not os.path.isfile(file_path):
                    print(f'Exception: Maze file does not exist. Unable to load saved maze!!')
                    exit(0)
                file_path = self.data_dir + self.params['LOCATION_FILENAME']
                if not os.path.isfile(file_path):
                    print(f'Exception: Source/Destination file does not exist. Unable to load saved maze!!')
                    exit(0)

        self.env = MazeEnv()
        self.agent = Agent(self.env)
        self.n_obs = self.params['NUM_OBSTACLE']

        if self.is_new_map:
            self.env.generate_maze(self.n_obs)
            np.save(self.data_dir + filename, self.env.maze)
            self.env.generate_src_dst()
        else:
            self.env.maze = np.load(self.data_dir + filename)
            self.env.load_src_dst()

        self.train_mode = train_mode
        self.is_trained = None
        self.is_test_completed = False
        self.render = render
        self.train_episodes = train_episodes
        self.test_episodes = self.params['TEST_EPISODES']

        # Value-based model selection (DQN / Double DQN / Dueling DQN)
        if args.dqn:
            self.agent.model_name = 'DQN'
        elif args.ddqn:
            self.agent.model_name = 'Double DQN'
        elif args.dueldqn:
            self.agent.model_name = 'Dueling DQN'
        else:
            print(f'Exception: No value-based model specified. Use --dqn, --ddqn, or --dueldqn.')
            exit(0)

        self.train_start_time = None
        self.train_end_time = None
        self.sim_start_time = None
        self.sim_end_time = None
        self.is_env_initialized = False
        self.running = True

    def run_simulation(self):
        self.game_initialize()
        self.sim_start_time = time.time()

        while self.running:
            self.event_on_game_window() if self.render else None
            if self.train_mode and not self.is_trained:
                print(f'=' * 65 + ' Training Phase ' + '=' * 66)
                self.train_start_time = time.time()

                result = self.agent.train_value_agent(self.train_episodes, self.render)

                train_data_visual = DataVisualization(self.train_episodes, result,
                                                      self.agent.model_name, 'VALUE')

                train_data_visual.save_data()
                train_data_visual.plot_returns()
                train_data_visual.plot_episode_length()
                train_data_visual.plot_training_error()
                train_data_visual.plot_epsilon_decay()

                self.train_end_time = time.time()
                elapsed_time = self.train_end_time - self.train_start_time
                print(f'Info: Training has been completed...')
                print(f'Info: Total Completion Time: {elapsed_time:.2f} seconds')
                print(f'-' * 147)
                self.is_trained = True

            if (self.is_trained or not self.train_mode) and not self.is_test_completed:
                break

    def game_initialize(self):
        self.agent.position = np.array(self.env.source)
        self.env.find_path()

        if self.render:
            self.env.env_setup()

        if self.train_mode:
            self.is_trained = False
        else:
            self.is_trained = True

        self.is_env_initialized = True
        print(f'Info: Source: {self.env.source} Destination: {self.env.destination}')
        print(f'-' * 147)
        # return len(self.env.find_path())

        if self.agent.model_name == 'DQN':
            print(f'Info: Model Selected: {self.agent.model_name}')
            self.agent.model = DQNModel(self.agent.state_size, self.agent.action_size, self.env, self.agent.device)
            print(f'Info: DQN Model is assigned for the Training and Testing of Agent...')
        elif self.agent.model_name == 'Double DQN':
            print(f'Info: Model Selected: {self.agent.model_name}')
            self.agent.model = DoubleDQNModel(self.agent.state_size, self.agent.action_size, self.env, self.agent.device)
            print(f'Info: DoubleDQ Model is assigned for the Training and Testing of Agent...')
        elif self.agent.model_name == 'Dueling DQN':
            print(f'Info: Model Selected: {self.agent.model_name}')
            self.agent.model = DuelingDQNModel(self.agent.state_size, self.agent.action_size, self.env, self.agent.device)
            print(f'Info: DuelingDQN Model is assigned for the Training and Testing of Agent...')


    def event_on_game_window(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def close_simulation(self):
        pygame.quit() if self.render else None

