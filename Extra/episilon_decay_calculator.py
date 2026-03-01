import matplotlib.pyplot as plt

def calculate_epsilon_decay(episodes, epsilon_start, epsilon_min, epsilon_decay):
    """
    Calculate epsilon values over episodes using exponential decay.

    Args:
        episodes (int): Total number of episodes.
        epsilon_start (float): Initial epsilon value.
        epsilon_min (float): Minimum epsilon value.
        epsilon_decay (float): Decay rate per episode.

    Returns:
        list: Epsilon values for each episode.
    """
    if episodes <= 0:
        raise ValueError("Number of episodes must be positive.")
    if not (0 < epsilon_min <= epsilon_start <= 1):
        raise ValueError("Epsilon values must be between 0 and 1, with start >= min.")
    if not (0 < epsilon_decay <= 1):
        raise ValueError("Epsilon decay rate must be between 0 and 1.")

    epsilon_values = []
    epsilon = epsilon_start

    for _ in range(episodes):
        epsilon_values.append(epsilon)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)  # Decay formula

    return epsilon_values


if __name__ == "__main__":
    # Parameters
    episodes = 10000
    epsilon_start = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9991  # Decay rate per episode

    # Calculate epsilon values
    epsilon_values = calculate_epsilon_decay(episodes, epsilon_start, epsilon_min, epsilon_decay)

    # Plot epsilon decay
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, episodes + 1), epsilon_values, marker='o', markersize=3, label="Epsilon Value")
    plt.axhline(y=epsilon_min, color='r', linestyle='--', label="Epsilon Min")
    plt.title("Epsilon Decay Over Episodes (DQN)")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
