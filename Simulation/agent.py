import json
import os
import torch
import numpy as np


class Agent:
    def __init__(self, env):
        param_dir = 'Simulation/Utils/'
        params = {}
        for cfg in ('env_params.json', 'train_params.json', 'sim_params.json'):
            with open(param_dir + cfg) as f:
                params.update(json.load(f))
        self.params = params

        self.model_save_path = self.params['DATA_DIR'] + self.params['MAZE_MODEL_DIR']
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.env = env
        self.position = None
        self.radius = self.params['RADIUS']
        self.model_name = None
        self.model = None
        self.state_size = self.env.maze_size * self.env.maze_size
        self.action_size = self.params['ACTION_SIZE']
        self.batch_size = self.params['BATCH_SIZE']
        self.model_filename = None
        self.game_steps = 0
        self.max_steps  = 200   # overwritten by Simulation after A* path is found
        self.epch = 1   # set to the current trial number by Simulation

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f'Info: GPU is available...')
        else:
            self.device = torch.device("cpu")
            print(f'Info: CPU is available...')

        self.visited_states = None
        self.max_no_progress_steps = 10
        self.no_progress_steps = 0
        self.last_distance_to_goal = np.inf

        self.success_episodes = []
        self.total_steps = []

        # Live display tracking (read by maze_env left panel)
        self.current_episode   = 0
        self.total_episodes    = 0
        self.episode_reward    = 0.0
        self.cumulative_reward = 0.0
        self.goal_count        = 0

        # Per-episode history lists — read by maze_env right panel for live sparklines
        self.live_rewards   = []   # episode return
        self.live_epsilons  = []   # epsilon after each episode
        self.live_steps     = []   # steps taken per episode
        self.live_errors    = []   # TD / training loss per episode
        self.live_path_eff  = []   # path efficiency (optimal / steps), 0 if failed
        self._optimal_path_len = 0  # set once path is known

        # Testing live history — updated after each test episode
        self.test_live_rewards = []   # test episode return
        self.test_live_steps   = []   # test steps per episode
        self.test_live_success = []   # test success (1/0) per episode

    def move(self, direction):
        self.position += direction

    def reset(self):
        self.position       = np.array(self.env.source)
        self.visited_states = set()
        self.game_steps     = 0   # must reset so every episode has a full step budget
        return self._get_state()

    def _get_state(self):
        agent_state_index = self.position[0] * self.env.maze_size + self.position[1]
        return agent_state_index

    def step(self, action):
        self.game_steps += 1
        terminated, truncated, info = False, False, {'Success': False}
        direction    = np.array((self.env.directions[action][0], self.env.directions[action][1]))
        new_position = [self.position[0] + direction[0], self.position[1] + direction[1]]

        if self.env.is_valid_position(new_position):
            self.move(direction)
            if np.array_equal(self.position, self.env.destination):
                print(f'\nInfo: HURRAY!! Agent has reached its destination...')
                reward = 5.0
                terminated = True
                self.game_steps = 0
                info['Success'] = True
            else:
                reward = 0.05
                if self.game_steps >= self.max_steps:
                    truncated = True
                    self.game_steps = 0
                else:
                    if tuple(self.position) not in self.visited_states:
                        reward += 0.05
                        self.visited_states.add(tuple(self.position))
                    else:
                        reward -= 0.07

                    current_dist_to_goal = self.env.distance_to_goal(self.position)
                    if current_dist_to_goal <= self.last_distance_to_goal:
                        reward += 0.05
                        self.last_distance_to_goal = current_dist_to_goal
                    else:
                        reward -= 0.10
        else:
            reward = -0.75
            terminated = True
            self.game_steps = 0
        return self._get_state(), reward, terminated, truncated, info

    def train_value_agent(self, episodes, render):
        print(f'Info: Agent Training has been started over the Maze Simulation...')
        print(f'-' * 147)

        # Filename mirrors the Excel naming: DQN_200_episode_60x60_simple_Trial_1_best.pt
        sz         = self.env.maze_size
        difficulty = self.params.get('DIFFICULTY', 'simple')
        self.model_filename = (
            f"{self.model_name}_{episodes}_episode_"
            f"{sz}x{sz}_{difficulty}_Trial_{self.epch}_best.pt"
        )

        returns_per_episode = np.zeros(episodes)
        epsilon_history = np.zeros(episodes)
        steps_per_episode = np.zeros(episodes)
        training_error = np.zeros(episodes)
        unique_steps_per_episode = np.zeros(episodes)
        success_paths = []

        self.total_episodes = episodes
        best_reward = float('-inf')   # track best episode return seen so far
        time_steps  = 0
        for episode in range(episodes):
            self.current_episode = episode
            self.episode_reward  = 0.0
            pth=[]
            state = self.reset()
            done, returns, step, success_status, loss = False, 0, 0, 0, 0.0
            while True:
                if render:
                    self.env.update_display(self)
                    if self.env.quit_requested:
                        done = True
                        break
                time_steps += 1
                pth.append(self.position.copy())

                if time_steps % self.model.update_rate == 0:
                    self.model.update_target_network()

                action = self.model.act(state)
                new_state, reward, terminated, truncated, info = self.step(action)

                self.model.remember(state, action, reward, new_state, terminated)

                state = new_state
                returns += reward
                self.episode_reward    = returns
                self.cumulative_reward += reward
                step += 1
                done = terminated or truncated

                if info['Success']:
                    self.goal_count += 1

                if done:
                    print(f"Episode {episode + 1}/{episodes} - Steps: {step}, Return: {returns:.2f}, Epsilon: "
                          f"{self.model.epsilon:.3f}, Loss: {loss:0.4f}")
                    self.total_steps.append(step)

                    if np.array_equal(self.position, self.env.destination):
                        self.success_episodes.append(episode)
                        success_paths.append(pth)
                        # Save only when this successful episode beats the best so far
                        if returns > best_reward:
                            best_reward = returns
                            self.save_model(is_policy_model=False)
                            print(f'Info: New best model saved  (reward={best_reward:.3f})')

                    break

                if len(self.model.replay_buffer.buffer) > self.batch_size:
                    loss = self.model.train(self.batch_size)

            if self.env.quit_requested:
                print('Info: Window closed — stopping training early.')
                break
            self.model.epsilon = max(self.model.epsilon * self.model.epsilon_decay, self.model.epsilon_min)
            unique_steps_per_episode[episode] = len(self.visited_states)
            returns_per_episode[episode] = returns
            epsilon_history[episode] = self.model.epsilon
            steps_per_episode[episode] = step
            training_error[episode] = loss

            # Update live sparkline histories (read by right panel each frame)
            opt = len(self.env.path) - 1 if (getattr(self.env, 'path', None)) else 1
            self._optimal_path_len = max(1, opt)
            pe = self._optimal_path_len / max(1, step) if episode in self.success_episodes else 0.0
            self.live_rewards.append(float(returns))
            self.live_epsilons.append(float(self.model.epsilon))
            self.live_steps.append(float(step))
            self.live_errors.append(float(loss))
            self.live_path_eff.append(float(pe))


        # storing the success paths and success episodes as json so later can be used to plot the heatmaps.
        optimal_steps = max(1, len(self.env.path) - 1) if hasattr(self.env, "path") and self.env.path else None
        optimal_path = self.env.path or []
        optimal_path_list = [[int(r), int(c)] for (r, c) in optimal_path]
        save_payload = {
            "model_name": self.model_name,
            "maze_size": int(self.env.maze_size),
            "source": self.env.source.tolist(),
            "destination": self.env.destination.tolist(),
            "train_episodes": int(episodes),
            "optimal_steps": optimal_steps,
            "success_episodes": list(map(int, self.success_episodes)),
            # Convert (r, c) tuples to [r, c] lists for JSON
            "success_paths": [[[int(r), int(c)] for (r, c) in path] for path in success_paths],
            "optimal_path": optimal_path_list,
        }

        filename = f"{self.model_name}_paths_{self.env.maze_size}x{self.env.maze_size}_{difficulty}_{episodes}_ep_trial_{self.epch}.json"
        with open(os.path.join(self.model_save_path, filename), "w") as f:
            json.dump(save_payload, f)
        print(f"Info: Saved success paths and episodes to {filename}")

        print(f'-' * 147)
        if best_reward == float('-inf'):
            # No success in the entire run — save final weights so testing still works
            self.save_model(is_policy_model=False)
            print('Info: No successful episode — saved final weights as fallback.')
        print(f'-' * 147)
        return [returns_per_episode, epsilon_history, training_error, steps_per_episode, self.success_episodes,unique_steps_per_episode,success_paths]

    def test_value_agent(self, episodes, render):
        print(f'Info: Testing of the Agent has been started over the Maze Simulation...')
        print(f'Info: Source: {self.env.source} Destination: {self.env.destination}')
        print(f'Info: Loading model → {self.model_filename}')
        print(f'-' * 147)

        if self.model_filename is None:
            print('Exception: model_filename is not set — cannot load weights.')
            exit(0)

        returns_per_episode = np.zeros(episodes)
        steps_per_episode   = np.zeros(episodes)
        success_per_episode = np.zeros(episodes)

        if os.path.exists(self.model_save_path):
            file_path = self.model_save_path + self.model_filename
            if os.path.isfile(file_path):
                self.model.main_network.load_state_dict(torch.load(file_path, weights_only=True))
                print(f'Info: Saved model has been successfully loaded...')
                print(f'-' * 147)
            else:
                print(f'Exception: Model file is not exists. Unable to load saved model weight!!')
                exit(0)
        else:
            print(f'Exception: The Data directory is not exists. Unable to load saved model weight!!')
            exit(0)

        self.model.main_network.eval()

        self.total_episodes    = episodes   # left-panel denominator
        self.cumulative_reward = 0.0        # reset for clean test display

        for episode in range(episodes):
            state = self.reset()
            self.current_episode = episode              # ← left panel: episode counter
            self.episode_reward  = 0.0                 # ← left panel: per-episode reward
            done, returns, step, success_status = False, 0, 0, 0
            while not done:
                if render:
                    self.env.update_display(self)
                    if self.env.quit_requested:
                        done = True
                        break
                with torch.no_grad():
                    action = self.model.main_network(
                        self.model.encode_state(state).to(self.device)
                    ).argmax().item()

                new_state, reward, terminated, truncated, info = self.step(action)
                state   = new_state
                step   += 1
                done    = terminated or truncated
                returns += reward
                self.episode_reward    = returns
                self.cumulative_reward += reward

                if info['Success']:
                    success_status = 1
                    self.goal_count += 1

            returns_per_episode[episode] = returns
            steps_per_episode[episode]   = step
            success_per_episode[episode] = success_status
            # Update right-panel sparklines
            self.test_live_rewards.append(float(returns))
            self.test_live_steps.append(float(step))
            self.test_live_success.append(float(success_status))
            print(f'Episode {episode + 1}/{episodes} - Steps: {step}, '
                  f'Return: {returns:.2f}, Success: {bool(success_status)}')

            if self.env.quit_requested:
                print('Info: Window closed — stopping testing early.')
                break

        success_pct = success_per_episode.mean() * 100
        print(f'-' * 147)
        print(f'Info: Testing completed — Success rate: {success_pct:.1f}%')
        return {
            'returns':  returns_per_episode,
            'steps':    steps_per_episode,
            'success':  success_per_episode,
        }

    def save_model(self, is_policy_model):
        if is_policy_model:
            torch.save(self.model.policy_network.state_dict(), self.model_save_path + self.model_filename)
        else:
            torch.save(self.model.main_network.state_dict(), self.model_save_path + self.model_filename)
        print(f'Info: The model has been saved...')
