import argparse
import json
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
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
        # result may be None when the instance is used only for save_test_data()
        if result is not None:
            self.returns               = result[0]
            self.epsilon_decay_history = result[1]
            self.training_error        = result[2]
            self.steps                 = result[3]
            self.success_episodes      = result[4]
            self.unique_steps          = result[5]
        else:
            self.returns = self.epsilon_decay_history = self.training_error = None
            self.steps   = self.success_episodes = self.unique_steps = None
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
        # Filename: e.g.  DQN_200_episode_60x60_simple_train.xlsx
        # Each trial is a separate sheet:  Trial_1, Trial_2, …
        fname    = (f"{self.model}_{self.n_episodes}_episode_"
                    f"{self.sz}x{self.sz}_{self.difficulty}_train.xlsx")
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

    def save_test_data(self, test_result: dict, test_episodes: int):
        """Save per-episode test results to Excel.

        Filename mirrors training:  DQN_50_episode_60x60_complex_test.xlsx
        Each trial → separate sheet: Trial_1, Trial_2, …
        """
        fname      = (f"{self.model}_{test_episodes}_episode_"
                      f"{self.sz}x{self.sz}_{self.difficulty}_test.xlsx")
        filepath   = self.fig_dir + fname
        sheet_name = f"Trial_{self.epch}"

        returns = test_result['returns']
        steps   = test_result['steps']
        success = test_result['success']

        df = pd.DataFrame({
            'Rewards':      returns,
            'Steps':        steps,
            'Success':      success.astype(int),
        })

        # Path efficiency: optimal / steps taken (0 for failed episodes)
        df['Path Efficiency'] = 0.0
        for ep in range(test_episodes):
            if success[ep]:
                df.loc[ep, 'Path Efficiency'] = self.optimal_path_length / max(1, steps[ep])

        df['Success Rate (%)'] = success.mean() * 100   # same value each row, convenient summary

        if not os.path.isfile(filepath):
            with pd.ExcelWriter(filepath, mode='w') as writer:
                df.to_excel(writer, sheet_name=sheet_name)
        else:
            with pd.ExcelWriter(filepath, mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=sheet_name)

        print(f'Info: Test data saved → {fname}  (sheet: {sheet_name})')

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

def perform_statistical_analysis(args, train_episodes: int, is_test: bool = False):
    """ Construct filepath from args and parameters, then perform statistical analysis. """
    # Load all parameters
    param_dir = 'Simulation/Utils/'
    params = {}
    for cfg in ('env_params.json', 'train_params.json', 'sim_params.json'):
        with open(param_dir + cfg) as f:
            params.update(json.load(f))
            
    # Resolve the model name
    from Simulation.maze_simulation import _MODEL_REGISTRY
    model_flag = next((k for k in _MODEL_REGISTRY if getattr(args, k, False)), None)
    if model_flag is None:
        print('Exception: No model specified for statistical analysis.')
        return
    model_name, _ = _MODEL_REGISTRY[model_flag]
    
    sz = params['SIZE']
    difficulty = params.get('DIFFICULTY', 'simple')
    fig_dir = params['DATA_DIR'] + params['FIG_DIR']
    
    if is_test:
        n_episodes = params.get('TEST_EPISODES', 50)
        mode_str = 'test'
    else:
        n_episodes = train_episodes
        mode_str = 'train'
        
    filename = f"{model_name}_{n_episodes}_episode_{sz}x{sz}_{difficulty}_{mode_str}.xlsx"
    filepath = fig_dir + filename
    
    if not os.path.exists(filepath):
        print(f"File not found for statistical analysis: {filepath}")
        return

    # Read all sheets
    try:
        all_sheets = pd.read_excel(filepath, sheet_name=None)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return
        
    trial_sheets = {name: df for name, df in all_sheets.items() if name.startswith('Trial_')}
    if not trial_sheets:
        print(f"No trial sheets found in {filepath}")
        return
        
    n_trials = len(trial_sheets)
    if n_trials < 2:
        print(f"Not enough trials ({n_trials}) in {filepath} for statistical analysis. Need at least 2.")
        return

    # Combine data from all trials into a 3D structure: [trial, episode, metric]
    # To keep it simple, we'll extract metric by metric
    metrics_to_analyze = ['Rewards', 'Steps', 'Path Efficiency', 'Success Rate']
    if not is_test:
        metrics_to_analyze.extend(['RGEE'])
        
    first_sheet = list(trial_sheets.values())[0]
    n_episodes = len(first_sheet)
    
    # Check what metrics are actually present
    available_metrics = [m for m in metrics_to_analyze if m in first_sheet.columns or m == 'Success Rate (%)']
    
    analysis_results = []
    
    for metric in available_metrics:
        col_name = metric
        if metric == 'Success Rate' and 'Success Rate (%)' in first_sheet.columns:
            col_name = 'Success Rate (%)'
            
        # Collect all values from all trials for this metric
        all_values = []
        for df in trial_sheets.values():
            if col_name in df.columns:
                # We extract the entire column
                all_values.extend(df[col_name].dropna().values.tolist())
                
        if not all_values:
            continue
            
        # Convert to numpy and filter out NaN or Infinite values
        values = np.array(all_values, dtype=float)
        values = values[np.isfinite(values)]
        
        if len(values) == 0:
            continue
            
        # Statistical calculations across all trials and all episodes combined
        n = len(values)
        mu = np.mean(values)
        sigma = np.std(values, ddof=1) if n > 1 else 0.0
        
        # 95% Confidence Interval using t-distribution
        if sigma > 0:
            t_crit = stats.t.ppf(0.975, df=n-1) # Two-tailed 95%
            margin_error = t_crit * (sigma / np.sqrt(n))
            ci_lower = mu - margin_error
            ci_upper = mu + margin_error
        else:
            ci_lower, ci_upper = mu, mu
            
        # mu +- 2*sigma
        bound_2_lower = mu - 2*sigma
        bound_2_upper = mu + 2*sigma
        
        # Null Hypothesis H0: mu = 0
        h0_rejected_ci = not (ci_lower <= 0 <= ci_upper)
        
        remark = f"Reject H0 (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])" if h0_rejected_ci else f"Accept H0 (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])"
        
        # Append data as one row per metric
        analysis_results.append({
            'Metric': metric,
            'Mean (mu)': mu,
            'Std (sigma)': sigma,
            'mu+-2sigma': f"[{bound_2_lower:.3f}, {bound_2_upper:.3f}]",
            'H0 Remark': remark
        })
        
    df_analysis = pd.DataFrame(analysis_results)
    
    # Save the analysis to the same Excel file in a new sheet
    try:
        with pd.ExcelWriter(filepath, mode='a', if_sheet_exists='replace') as writer:
            df_analysis.to_excel(writer, sheet_name='Statistical_Analysis', index=False)
        print(f"Statistical analysis appended to {filepath}")
    except Exception as e:
        print(f"Error writing statistical analysis to {filepath}: {e}")
