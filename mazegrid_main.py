"""
Main driver for the Deep RL-Based Maze Solver simulation.
Runs multiple training iterations and persists epoch count to sim_params.json.

Parameter Files (under Simulation/Utils/):
    env_params.json   - Maze environment settings (size, obstacles, FPS, etc.)
    train_params.json - DQN training hyperparameters (gamma, epsilon, batch size, etc.)
    sim_params.json   - Simulation I/O settings (paths, test episodes, epoch counter)
"""

from pathlib import Path

from Simulation.Utils.utils import setup_parser, perform_statistical_analysis
from Simulation.maze_simulation import Simulation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_RUNS       = 2       
TRAIN_EPISODES = 200
TRAIN_MODE     = True
RENDER         = True
PARAM_DIR      = Path('Simulation/Utils')


def main(args):
    """Run the maze simulation for NUM_RUNS iterations, updating epoch count each run."""
    for trial in range(1, NUM_RUNS + 1):
        sim = Simulation(args, TRAIN_MODE, TRAIN_EPISODES, RENDER, trial=trial)
        sim.run_simulation()
        sim.close_simulation()

    # Perform statistical analysis over the trials
    if TRAIN_MODE:
        print("\nPerforming statistical analysis on training data...")
        perform_statistical_analysis(args, TRAIN_EPISODES, is_test=False)
        
    print("\nPerforming statistical analysis on testing data...")
    perform_statistical_analysis(args, TRAIN_EPISODES, is_test=True)


if __name__ == '__main__':
    args = setup_parser()
    main(args)
