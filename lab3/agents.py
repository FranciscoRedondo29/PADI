import random
import pickle

"""
Agent classes for the fishing game.
Students should create their own agent classes by inheriting from the base Agent class.
"""


class Agent:
    """
    Base class for all fishing game agents.
    Students should inherit from this class and implement their own strategies.
    """

    def __init__(self):
        """Initialize the agent. Override this method if you need to set up any state."""
        pass

    def get_action(self, state):
        """
        Decide what action to take given the current game state.

        Args:
            state (dict): Dictionary containing:
                - 'fish_y': Y position of the fish
                - 'bar_y': Y position of the green bar
                - 'bar_vel': Velocity of the bar
                - 'catch_timer': Current catch timer value

        Returns:
            bool: True to thrust upward, False to let gravity pull down
        """
        pass

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        Optional: Update internal state/policy based on experience.

        Args:
            state:       The state in which the action was taken
            action:      The action that was taken
            reward:      Reward received after taking the action
            next_state:  The state observed after the action
            next_action: The action chosen in next_state (used by on-policy methods like SARSA)
            done:        Whether the episode has ended
        """
        pass

    def end_episode(self):
        """
        Optional: Called at the end of each episode.
        """
        pass

    def set_training_mode(self, training):
        """
        Optional: Switch between training (exploration) and testing (exploitation) modes.
        """
        pass

    def save_q_table(self, filename):
        """
        Optional: Save Q-table to file.
        """
        pass

    def load_q_table(self, filename):
        """
        Optional: Load Q-table from file.
        """
        pass


class PredictiveAgent(Agent):
    """
    Example: An agent that tries to predict where the fish will be.
    This considers the bar's velocity to make smoother movements.
    """

    def __init__(self, reaction_distance=20):
        """
        Initialize the predictive agent.

        Args:
            reaction_distance: How far ahead to start reacting (in pixels)
        """
        super().__init__()
        self.reaction_distance = reaction_distance

    def get_action(self, state):
        """
        Thrust if fish is above the bar center, accounting for reaction distance.

        Args:
            state (dict): Current game state

        Returns:
            bool: True to thrust, False to fall
        """
        bar_center = state["bar_y"] + 40  # Bar is 80 pixels tall, so center is +40

        # If fish is above bar center (with buffer), thrust
        if state["fish_y"] < bar_center - self.reaction_distance:
            return True
        return False


class TDAgent(Agent):
    """
    Base class for tabular Temporal-Difference agents (Q-Learning, SARSA, …).

    Provides the shared machinery:
      - Hyperparameters (alpha, gamma, epsilon / decay)
      - Q-table storage and lookup
      - Epsilon-greedy action selection
      - State discretization
      - Episode bookkeeping (epsilon decay, episode counter)
      - Training-mode toggle
      - Q-table save / load

    Derived classes only need to implement learn().
    """

    def __init__(
        self,
        alpha=0.1,  # Learning rate
        gamma=0.99,  # Discount factor
        epsilon=0.1,  # Exploration rate
        epsilon_decay=0.995,  # Epsilon decay per episode
        epsilon_min=0.01,  # Minimum epsilon
    ):
        """
        Initialize the TD agent.

        Args:
            alpha:         Learning rate (how much to update Q-values)
            gamma:         Discount factor (how much to value future rewards)
            epsilon:       Initial exploration rate (probability of random action)
            epsilon_decay: Factor to decay epsilon after each episode
            epsilon_min:   Minimum epsilon value
        """
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: maps (discrete_state, action) -> Q-value
        self.q_table = {}

        # Training mode flag
        self.training = True

        # Statistics
        self.episodes_trained = 0

    # ------------------------------------------------------------------
    # State discretization
    # ------------------------------------------------------------------

    def discretize_state(self, state):
        """
        Convert continuous state to discrete bins for the Q-table.

        Args:
            state (dict[str, float]): Continuous game state

        Returns:
            tuple[int|float, ...]: Discretized state as a hashable tuple
        """
        # 20 pixels for positions, 2 units for velocity
        fish_bin = int(state['fish_y'] // 20)
        bar_bin = int(state['bar_y'] // 20)
        vel_bin = int(state['bar_vel'] // 2)
        
        return (fish_bin, bar_bin, vel_bin)

    # ------------------------------------------------------------------
    # Q-value helpers
    # ------------------------------------------------------------------

    def get_q_value(self, state, action):
        """
        Get Q-value for a (discrete) state-action pair.

        Args:
            state:  Discretized state tuple
            action: Action (True or False)

        Returns:
            float: Q-value (0.0 if not seen before)
        """
        return self.q_table.get((state, action), 0.0)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def get_action(self, state):
        """
        Choose action, using epsilon-greedy strategy.

        Args:
            state (dict[str, float]): Original game state (not yet discretized).

        Returns:
            bool: True to thrust, False to fall
        """
        # 1. Exploration: Take a random action with probability epsilon
        if self.training and random.random() < self.epsilon:
            return random.choice([True, False])
            
        # 2. Exploitation: Choose the best action based on Q-values
        discrete_state = self.discretize_state(state)
        
        q_thrust = self.get_q_value(discrete_state, True)
        q_fall = self.get_q_value(discrete_state, False)
        
        # Pick the action with the highest Q-value
        if q_thrust > q_fall:
            return True
        elif q_fall > q_thrust:
            return False
        else:
            # If Q-values are equal (e.g., at the start of training), break ties randomly
            return random.choice([True, False])

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------

    def end_episode(self):
        """Decay epsilon and increment episode counter."""
        self.episodes_trained += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Training mode
    # ------------------------------------------------------------------

    def set_training_mode(self, training):
        """
        Set whether the agent is in training or evaluation mode.

        Args:
            training (bool): If True, use epsilon-greedy. If False, always exploit.
        """
        self.training = training

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_q_table(self, filepath):
        """Save Q-table to file."""
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "episodes_trained": self.episodes_trained,
                    "epsilon": self.epsilon,
                },
                f,
            )
        print(f"Q-table saved to {filepath}")

    def load_q_table(self, filepath):
        """Load Q-table from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.q_table = data["q_table"]
            self.episodes_trained = data.get("episodes_trained", 0)
            self.epsilon = data.get("epsilon", self.epsilon)
        print(
            f"Q-table loaded from {filepath} ({len(self.q_table)} entries, {self.episodes_trained} episodes)"
        )


class QLearningAgent(TDAgent):
    """
    Tabular Q-Learning agent (off-policy TD control).

    Update rule:

    The max over next actions makes this off-policy: it always learns towards
    the greedy policy regardless of which action is actually taken next.
    """

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        Update Q-table using the Q-learning (off-policy) rule.

        Args:
            state:       State in which the action was taken (dict)
            action:      Action that was taken
            reward:      Reward received
            next_state:  State observed after the action (dict)
            next_action: Action chosen in next_state (ignored by Q-learning)
            done:        Whether the episode has ended
        """
        if not self.training:
            return

        # FLIP THE COST TO A REWARD (-1 cost becomes +1 reward)
        actual_reward = -reward 

        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)

        current_q = self.get_q_value(discrete_state, action)

        if done:
            max_next_q = 0.0 
        else:
            q_thrust = self.get_q_value(discrete_next_state, True)
            q_fall = self.get_q_value(discrete_next_state, False)
            max_next_q = max(q_thrust, q_fall)

        # Use actual_reward instead of reward
        new_q = current_q + self.alpha * (actual_reward + self.gamma * max_next_q - current_q)
        
        self.q_table[(discrete_state, action)] = new_q


class SarsaLearningAgent(TDAgent):
    """
    SARSA Agent (on-policy TD control).

    Update rule:

    Uses the action actually chosen in the next state (a'), making this
    on-policy: the learned values reflect the agent's own behaviour policy.
    """

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        Update Q-table using the SARSA (on-policy) rule.

        Args:
            state:       State in which the action was taken (dict)
            action:      Action that was taken
            reward:      Reward received
            next_state:  State observed after the action (dict)
            next_action: Action chosen in next_state (used by SARSA)
            done:        Whether the episode has ended
        """
        if not self.training:
            return

        # FLIP THE COST TO A REWARD
        actual_reward = -reward 

        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)

        current_q = self.get_q_value(discrete_state, action)

        if done:
            next_q = 0.0 
        else:
            next_q = self.get_q_value(discrete_next_state, next_action)

        # Use actual_reward instead of reward
        new_q = current_q + self.alpha * (actual_reward + self.gamma * next_q - current_q)
        
        self.q_table[(discrete_state, action)] = new_q