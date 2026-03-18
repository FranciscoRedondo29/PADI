"""
Convergence Analysis Runner
Trains an agent with multiple random seeds and plots
mean ± std-dev of cumulative costs for training and testing.

Usage:
  python runner.py                   # defaults to SARSA
  python runner.py --agent sarsa
  python runner.py --agent qlearning
  python runner.py --agent predictive
"""

import argparse
import random
import numpy as np

from fishing_logic import FishingGameLogic, FISH_TYPES
from agents import PredictiveAgent, QLearningAgent, SarsaLearningAgent

AGENT_REGISTRY = {
    "sarsa":      ("SARSA",          SarsaLearningAgent),
    "qlearning":  ("Q-Learning",     QLearningAgent),
    "predictive": ("PredictiveAgent", PredictiveAgent),
}


def run_agent(
    agent,
    fish_types=None,
    num_episodes=2500,
    do_learning=True,
    verbose=True,
    visualize=False,
):
    """
    Run the agent in the environment for a set number of episodes.
    If do_learning=True, the agent will explore and update its policy.
    If do_learning=False, the agent will strictly exploit its current policy.
    """
    assert fish_types or num_episodes

    if fish_types:
        num_episodes = len(fish_types)

    mode_str = "Training" if do_learning else "Testing"

    # Handle agent state for testing
    if not do_learning and hasattr(agent, "set_training_mode"):
        agent.set_training_mode(False)
    elif verbose:
        print(f"\n{'=' * 60}")
        print(f"{mode_str} {agent.__class__.__name__}")
        print(f"{'=' * 60}")

    visualizer = None

    if visualize:
        from visualize import GameVisualizer

        visualizer = GameVisualizer(
            f"{agent.__class__.__name__} ({mode_str})", "Random"
        )

    wins = 0
    total_cost = 0
    total_steps = 0
    costs_history = []

    for episode in range(num_episodes):
        game = FishingGameLogic(fish_name=fish_types[episode] if fish_types else None)
        done = False
        episode_cost = 0
        steps = 0

        while not done and steps < 2500:
            state = game.get_state()
            action = agent.get_action(state)

            next_state, cost, done = game.step_physics(action)
            next_action = agent.get_action(next_state)

            if visualizer:
                visualizer.update(game.get_state(), done)

            if do_learning:
                agent.learn(state, action, cost, next_state, next_action, done)

            episode_cost += cost
            steps += 1

        if do_learning:
            agent.end_episode()

        costs_history.append(episode_cost)
        total_cost += episode_cost
        total_steps += steps

        if game.catch_timer > 0:
            wins += 1

        # Print progress only when training and verbose
        if do_learning and verbose and (episode + 1) % 100 == 0:
            win_rate = wins / (episode + 1) * 100

            epsilon = getattr(agent, "epsilon", "N/A")
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Win Rate: {win_rate:5.1f}% | "
                f"ε: {epsilon if epsilon == 'N/A' else f'{epsilon:.3f}'}"
            )

    # Re-enable training if it was disabled for testing
    if not do_learning and hasattr(agent, "set_training_mode"):
        agent.set_training_mode(True)

    if visualizer:
        import time

        time.sleep(1)
        visualizer.close()

    win_rate = wins / num_episodes * 100
    avg_cost = total_cost / num_episodes
    avg_steps = total_steps / num_episodes

    return {
        "wins": wins,
        "win_rate": win_rate,
        "avg_cost": avg_cost,
        "avg_steps": avg_steps,
        "costs_history": costs_history,
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Convergence Analysis Runner")
    parser.add_argument(
        "--agent",
        choices=list(AGENT_REGISTRY.keys()),
        default=None,
        help="Agent to evaluate. Omit to run all agents.",
    )
    args = parser.parse_args()

    # Resolve which agents to run
    agents_to_run = (
        list(AGENT_REGISTRY.items())
        if args.agent is None
        else [(args.agent, AGENT_REGISTRY[args.agent])]
    )

    # --- Configuration ---
    NUM_RUNS = 10              # Number of independent training runs (one per seed)
    NUM_TRAIN_EPISODES = 5000  # Episodes per training run
    SEEDS = [42 * (i + 1) for i in range(NUM_RUNS)]  # Deterministic, distinct seeds

    # Fixed test suite: 50 episodes per fish type, same order every run
    test_fish_types = [fish.name for fish in FISH_TYPES for _ in range(50)]

    # Collect results per agent
    all_agents_train = {}   # agent_key -> (runs, episodes) matrix
    all_agents_test  = {}   # agent_key -> (runs, test_episodes) matrix
    all_agents_stats = {}   # agent_key -> list of per-run result dicts

    for agent_key, (agent_label, AgentClass) in agents_to_run:
        print(f"\n{'*' * 60}")
        print(f"{agent_label} Convergence Analysis — {NUM_RUNS} runs x {NUM_TRAIN_EPISODES} episodes")
        print(f"{'*' * 60}")

        train_costs_all = []
        test_costs_all  = []
        run_results     = []

        for run, seed in enumerate(SEEDS):
            print(f"\n--- {agent_label} Run {run + 1}/{NUM_RUNS}  (seed={seed}) ---")

            # Set both stdlib and numpy seeds for full reproducibility
            random.seed(seed)
            np.random.seed(seed)

            agent = AgentClass()

            # --- Train ---
            train_stats = run_agent(
                agent,
                fish_types=None,
                num_episodes=NUM_TRAIN_EPISODES,
                do_learning=True,
                verbose=True,
            )
            train_costs_all.append(np.cumsum(train_stats["costs_history"]))

            # --- Test (no exploration) ---
            test_stats = run_agent(
                agent,
                fish_types=test_fish_types,
                num_episodes=None,
                do_learning=False,
                verbose=(run == 0),
                visualize=False,
            )
            test_costs_all.append(np.cumsum(test_stats["costs_history"]))
            run_results.append(test_stats)

        all_agents_train[agent_key] = np.array(train_costs_all)
        all_agents_test[agent_key]  = np.array(test_costs_all)
        all_agents_stats[agent_key] = run_results

    # ------------------------------------------------------------------ #
    # Summary statistics
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 60}")
    print(f"{'Agent':<20} {'Win Rate Mean':>14} {'Win Rate Std':>13} {'CV %':>7}")
    print("-" * 56)
    for agent_key, (agent_label, _) in agents_to_run:
        run_results = all_agents_stats[agent_key]
        win_rates = [r["win_rate"] for r in run_results]
        avg_costs = [r["avg_cost"] for r in run_results]
        avg_steps = [r["avg_steps"] for r in run_results]
        cv = np.std(win_rates) / np.mean(win_rates) * 100 if np.mean(win_rates) > 0 else float("nan")
        print(f"{agent_label:<20} {np.mean(win_rates):>13.1f}% {np.std(win_rates):>12.2f}% {cv:>6.1f}%")

        robustness = (
            "Very robust" if cv < 5
            else "Moderately robust" if cv < 15
            else "High variance"
        )
        print(f"  → {robustness}")

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    print("\nGenerating Cumulative Cost Plots (Mean +/- 1 Std Dev)...")

    COLORS = ["steelblue", "darkorange", "seagreen", "crimson"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    title_agents = ", ".join(label for _, (label, _) in agents_to_run)
    fig.suptitle(
        f"Convergence Analysis: {title_agents} ({NUM_RUNS} runs, {NUM_TRAIN_EPISODES} train episodes)",
        fontsize=13, fontweight="bold"
    )

    ax1, ax2 = axes
    episodes = np.arange(1, NUM_TRAIN_EPISODES + 1)

    for (agent_key, (agent_label, _)), color in zip(agents_to_run, COLORS):
        train_matrix = all_agents_train[agent_key]
        test_matrix  = all_agents_test[agent_key]

        # Training
        train_mean = np.mean(train_matrix, axis=0)
        train_std  = np.std(train_matrix,  axis=0)
        ax1.plot(episodes, train_mean, color=color, linewidth=1.5, label=agent_label)
        ax1.fill_between(episodes, train_mean - train_std, train_mean + train_std,
                         color=color, alpha=0.2)
        for run_curve in train_matrix:
            ax1.plot(episodes, run_curve, color=color, linewidth=0.3, alpha=0.25)

        # Testing
        test_mean = np.mean(test_matrix, axis=0)
        test_std  = np.std(test_matrix,  axis=0)
        test_ep   = np.arange(1, len(test_mean) + 1)
        ax2.plot(test_ep, test_mean, color=color, linewidth=1.5, label=agent_label)
        ax2.fill_between(test_ep, test_mean - test_std, test_mean + test_std,
                         color=color, alpha=0.2)
        for run_curve in test_matrix:
            ax2.plot(test_ep, run_curve, color=color, linewidth=0.3, alpha=0.25)

    for ax, title in [(ax1, "Training: Cumulative Cost"), (ax2, "Testing: Cumulative Cost")]:
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Cost")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_filename = (
        f"{args.agent}_convergence.png" if args.agent
        else "all_agents_convergence.png"
    )
    plt.savefig(out_filename, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_filename}")
    plt.show()
