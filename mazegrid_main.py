"""
Main driver for the Deep RL-Based Maze Solver simulation.
Runs multiple training iterations and persists epoch count to sim_params.json.

Parameter Files (under Simulation/Utils/):
    env_params.json   - Maze environment settings (size, obstacles, FPS, etc.)
    train_params.json - DQN training hyperparameters (gamma, epsilon, batch size, etc.)
    sim_params.json   - Simulation I/O settings (paths, test episodes, epoch counter)
"""

import json
from pathlib import Path

from Simulation.Utils.utils import setup_parser
from Simulation.maze_simulation import Simulation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_RUNS = 5
TRAIN_EPISODES = 200
TRAIN_MODE = True
RENDER = True
PARAM_DIR = Path('Simulation/Utils')


def main(args):
    """Run the maze simulation for NUM_RUNS iterations, updating epoch count each run."""
    sim_config_path = PARAM_DIR / 'sim_params.json'

    with open(sim_config_path, 'r') as f:
        sim_params = json.load(f)

    for trial in range(1, NUM_RUNS + 1):
        sim = Simulation(args, TRAIN_MODE, TRAIN_EPISODES, RENDER, trial=trial)
        sim.run_simulation()
        sim.close_simulation()


if __name__ == '__main__':
    args = setup_parser()
    main(args)
