"""
Simulation orchestrator for the Deep RL-Based Maze Solver.

Responsibilities:
    - Load the correct pre-built maze from Simulation/grid_map/ based on
      SIZE and DIFFICULTY from the parameter files.
    - Initialise the environment, agent, and chosen value-based RL model.
    - Run training (and optionally testing) phases.
    - Manage the pygame window lifecycle.

Public API (preserved):
    run_simulation()        -- main entry point
    game_initialize()       -- env / model / display setup
    event_on_game_window()  -- pygame event loop
    close_simulation()      -- teardown
"""

import json
import os
import time

import numpy as np
import pygame

from .agent import Agent
from .envs.maze_env import MazeEnv
from .model.dqntorch import DQNModel, DoubleDQNModel, DuelingDQNModel
from .Utils.utils import DataVisualization

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIVIDER = '-' * 147
HEADER  = '%' * 50 + ' Deep Reinforcement Learning Based MAZE Solver ' + '%' * 50

# Maps CLI flag → (display name, model class)
_MODEL_REGISTRY = {
    'dqn':     ('DQN',         DQNModel),
    'ddqn':    ('Double DQN',  DoubleDQNModel),
    'dueldqn': ('Dueling DQN', DuelingDQNModel),
}


class Simulation:
    """Orchestrates the full RL maze simulation."""

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------
    def __init__(self, args, train_mode: bool,
                 train_episodes: int = 100, render: bool = False,
                 trial: int = 1):

        self.params = self._load_params()
        print(f'\n{HEADER}')

        # Resolve the value-based model from CLI flags
        model_flag = next((k for k in _MODEL_REGISTRY if getattr(args, k, False)), None)
        if model_flag is None:
            print('Exception: No model specified. Use --dqn, --ddqn, or --dueldqn.')
            exit(0)
        model_name, model_cls = _MODEL_REGISTRY[model_flag]

        # Environment and agent
        self.env   = MazeEnv()
        self.agent = Agent(self.env)
        self.agent.model_name = model_name
        self._model_cls       = model_cls

        # Load map: new (generated) or saved (from grid_map)
        if getattr(args, 'newmaze', False):
            print('Info: Simulation started with a newly generated maze')
            self._load_generated_maze()
        else:
            size       = self.params['SIZE']
            difficulty = self.params.get('DIFFICULTY', 'simple')
            print(f'Info: Simulation started with saved map  '
                  f'({size}x{size}, {difficulty})')
            self._load_saved_maze(size, difficulty)

        # Runtime flags
        self.train_mode        = train_mode
        self.train_episodes    = train_episodes
        self.test_episodes     = self.params.get('TEST_EPISODES', 50)
        self.render            = render
        self.running           = True
        self.is_trained        = None
        self.is_test_completed = False
        self._display_open     = False

        # Trial number (1 → NUM_RUNS) passed from main — used for display & filenames
        self.agent.epch        = trial

        # Timing
        self.sim_start_time = None

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _load_params() -> dict:
        """Read and merge all three parameter JSON files."""
        param_dir = 'Simulation/Utils/'
        params = {}
        for cfg in ('env_params.json', 'train_params.json', 'sim_params.json'):
            with open(param_dir + cfg) as f:
                params.update(json.load(f))
        return params

    def _load_generated_maze(self):
        """Generate a new random maze and persist it to Data/Maze/."""
        data_dir = self.params['DATA_DIR'] + self.params['MAZE_DIR']
        os.makedirs(data_dir, exist_ok=True)
        self.env.generate_maze(self.params.get('NUM_OBSTACLE', 0.2))
        np.save(data_dir + self.params['MAZE_FILENAME'], self.env.maze)
        self.env.generate_src_dst()

    def _load_saved_maze(self, size: int, difficulty: str):
        """Load pre-built maze and src/dst from Simulation/grid_map/."""
        grid_dir = self.params.get('GRID_MAP_DIR', 'Simulation/grid_map/')
        prefix   = f'{size}x{size}_{difficulty}'
        maze_path = os.path.join(grid_dir, f'{prefix}_maze.npy')
        loc_path  = os.path.join(grid_dir, f'{prefix}_src_dst.npy')

        for path, label in ((maze_path, 'Maze map'), (loc_path, 'Source/Destination')):
            if not os.path.isfile(path):
                print(f'Exception: {label} file not found → {path}')
                exit(0)

        self.env.maze = np.load(maze_path)
        location = np.load(loc_path)
        self.env.source, self.env.destination = location[0], location[1]

    def _init_model(self):
        """Instantiate the chosen RL model and attach it to the agent."""
        self.agent.model = self._model_cls(
            self.agent.state_size,
            self.agent.action_size,
            self.env,
            self.agent.device,
        )
        print(f'Info: Model Selected  : {self.agent.model_name}')
        print(f'Info: {self.agent.model_name} model ready for '
              f'{"Training" if self.train_mode else "Testing"}')

    def _save_and_plot(self, result):
        """Persist training results to Excel and generate plots."""
        vis = DataVisualization(
            self.train_episodes, result, self.agent.model_name, 'VALUE',
            trial=self.agent.epch          # ← pass trial number so each run → own sheet
        )
        vis.save_data()
        vis.plot_returns()
        vis.plot_episode_length()
        vis.plot_training_error()
        vis.plot_epsilon_decay()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def run_simulation(self):
        """Main simulation entry point — initialise, train, (test), then close."""
        self.game_initialize()
        self.sim_start_time = time.time()

        while self.running:
            self.event_on_game_window() if self.render else None

            # --- Training phase ---
            if self.train_mode and not self.is_trained:
                print('=' * 65 + ' Training Phase ' + '=' * 66)
                self.env.mode = 'TRAINING'

                t0     = time.time()
                result = self.agent.train_value_agent(self.train_episodes, self.render)
                self._save_and_plot(result)

                elapsed = time.time() - t0
                print(f'Info: Training completed in {elapsed:.2f} s')
                print(DIVIDER)
                self.is_trained = True

            # --- Testing phase (placeholder — extend here) ---
            if (self.is_trained or not self.train_mode) and not self.is_test_completed:
                break   # exit loop; testing logic to be added here

    def game_initialize(self):
        """Set up agent position, A* path, pygame display, and RL model."""
        self.agent.position = np.array(self.env.source)
        self.env.find_path()

        if self.render:
            self.env.env_setup()
            self._display_open = True

        self.is_trained = False if self.train_mode else True

        print(f'Info: Source      : {self.env.source}')
        print(f'Info: Destination : {self.env.destination}')
        print(DIVIDER)

        self._init_model()
        print(DIVIDER)

    def event_on_game_window(self):
        """Process pygame window events (e.g. close button)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def close_simulation(self):
        """Tear down pygame display if it is open."""
        if self._display_open:
            pygame.quit()
            self._display_open = False
