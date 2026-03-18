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
    parser.add_argument(
        "--experiment",
        choices=["epsilon", "decay", "single_fish"],
        default=None,
        help=(
            "Run exploration parameter experiments. "
            "'epsilon' compares different initial epsilon values; "
            "'decay' compares different epsilon_decay rates; "
            "'single_fish' trains on one fish type and tests on all."
        ),
    )
    args = parser.parse_args()

    # ================================================================== #
    # EXPERIMENT MODE  (--experiment epsilon | decay)
    # ================================================================== #
    if args.experiment:

        # --- Shared settings ---
        NUM_RUNS           = 5
        NUM_TRAIN_EPISODES = 5000
        SEEDS              = [42 * (i + 1) for i in range(NUM_RUNS)]
        test_fish_types    = [fish.name for fish in FISH_TYPES for _ in range(50)]

        # ---- Experiment configurations --------------------------------
        #
        # (a) epsilon: vary the initial exploration rate.
        #     epsilon_min is set equal to epsilon so the decay is visible
        #     only when epsilon > epsilon_min (High-ε config).
        #     Baseline keeps the original stuck-at-0.1 behaviour.
        #
        # (b) decay:   fix epsilon=1.0, epsilon_min=0.01 and vary the
        #              per-episode decay multiplier.
        # ---------------------------------------------------------------

        if args.experiment == "epsilon":
            EXPERIMENT_TITLE = "Experiment (a): Impact of Initial Epsilon"
            OUT_FILE         = "experiment_epsilon.png"
            configs = [
                dict(label="Baseline (ε=0.10, no decay)", color="steelblue",
                     epsilon=0.1,  epsilon_min=0.1,  epsilon_decay=0.995),
                dict(label="Low-ε  (ε=0.01)",            color="darkorange",
                     epsilon=0.01, epsilon_min=0.01, epsilon_decay=0.995),
                dict(label="High-ε (ε=1.00 → 0.01)",     color="seagreen",
                     epsilon=1.0,  epsilon_min=0.01, epsilon_decay=0.995),
            ]
        elif args.experiment == "decay":
            EXPERIMENT_TITLE = "Experiment (b): Impact of Epsilon Decay Rate"
            OUT_FILE         = "experiment_decay.png"
            # All three start at ε=1.0 → 0.01 so decay differences are visible
            configs = [
                dict(label="Fast  decay (d=0.990)",   color="crimson",
                     epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.990),
                dict(label="Medium decay (d=0.995)",  color="steelblue",
                     epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995),
                dict(label="Slow  decay (d=0.999)",   color="seagreen",
                     epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999),
            ]

        else:  # single_fish
            TRAIN_FISH       = "Sturgeon"
            EXPERIMENT_TITLE = f"Experiment (c): Single-Fish Training ({TRAIN_FISH}) vs. Mixed Training"
            OUT_FILE         = "experiment_single_fish.png"
            configs = [
                dict(
                    label="Original (mixed training)",
                    color="steelblue",
                    epsilon=0.1, epsilon_min=0.1, epsilon_decay=0.995,
                    train_fish=None,
                ),
                dict(
                    label=f"Single fish ({TRAIN_FISH} only)",
                    color="darkorange",
                    epsilon=0.1, epsilon_min=0.1, epsilon_decay=0.995,
                    train_fish=TRAIN_FISH,
                ),
            ]

        # ---- Resolve which agent class to use for experiments --------
        if args.agent:
            _exp_label, AgentClass = AGENT_REGISTRY[args.agent]
        else:
            AgentClass = QLearningAgent  # default for experiments

        # ---- Run all configurations -----------------------------------
        results = []   # list of dicts, one per config

        for cfg in configs:
            print(f"\n{'*' * 60}")
            print(f"{cfg['label']}  —  {NUM_RUNS} runs x {NUM_TRAIN_EPISODES} episodes")
            print(f"  epsilon={cfg['epsilon']}, epsilon_min={cfg['epsilon_min']}, "
                  f"epsilon_decay={cfg['epsilon_decay']}")
            print(f"{'*' * 60}")

            train_costs_all = []
            test_costs_all  = []
            run_avg_steps   = []

            for run, seed in enumerate(SEEDS):
                print(f"\n--- Run {run + 1}/{NUM_RUNS}  (seed={seed}) ---")
                random.seed(seed)
                np.random.seed(seed)

                agent = AgentClass(
                    epsilon=cfg["epsilon"],
                    epsilon_min=cfg["epsilon_min"],
                    epsilon_decay=cfg["epsilon_decay"],
                )

                # Determine which fish to train on (None = random mix)
                train_fish = cfg.get("train_fish")
                train_fish_list = (
                    [train_fish] * NUM_TRAIN_EPISODES if train_fish else None
                )
                train_stats = run_agent(
                    agent,
                    fish_types=train_fish_list,
                    num_episodes=NUM_TRAIN_EPISODES if train_fish_list is None else None,
                    do_learning=True,
                    verbose=True,
                )
                train_costs_all.append(np.cumsum(train_stats["costs_history"]))

                test_stats = run_agent(
                    agent,
                    fish_types=test_fish_types,
                    num_episodes=None,
                    do_learning=False,
                    verbose=False,
                )
                test_costs_all.append(np.cumsum(test_stats["costs_history"]))
                run_avg_steps.append(test_stats["avg_steps"])

            results.append({
                **cfg,
                "train_matrix": np.array(train_costs_all),
                "test_matrix":  np.array(test_costs_all),
                "avg_steps":    run_avg_steps,
            })

        # ---- Summary table -------------------------------------------
        print(f"\n{'=' * 75}")
        print(f"{'Config':<30} {'Win Rate':>10} {'Avg Steps':>12} {'Test Final Cost':>16}")
        print("-" * 75)
        for r in results:
            test_final = r["test_matrix"][:, -1]   # cumulative cost at last test episode
            # recompute win rate from avg_steps proxy not stored; use final cost as proxy
            print(f"{r['label']:<30} "
                  f"{'N/A':>10} "
                  f"{np.mean(r['avg_steps']):>12.1f} "
                  f"{np.mean(test_final):>16.1f}")

        # ---- Plots ---------------------------------------------------
        print(f"\nGenerating plots for {EXPERIMENT_TITLE}...")

        episodes  = np.arange(1, NUM_TRAIN_EPISODES + 1)
        test_ep   = np.arange(1, results[0]["test_matrix"].shape[1] + 1)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(
            f"{EXPERIMENT_TITLE}\n"
            f"{AgentClass.__name__} — {NUM_RUNS} runs × {NUM_TRAIN_EPISODES} train episodes",
            fontsize=13, fontweight="bold",
        )
        ax_train, ax_test, ax_steps = axes

        bar_x     = np.arange(len(results))
        bar_width = 0.5

        for r in results:
            color = r["color"]
            label = r["label"]

            # Training cumulative cost
            tr_mean = np.mean(r["train_matrix"], axis=0)
            tr_std  = np.std(r["train_matrix"],  axis=0)
            ax_train.plot(episodes, tr_mean, color=color, linewidth=1.5, label=label)
            ax_train.fill_between(episodes, tr_mean - tr_std, tr_mean + tr_std,
                                  color=color, alpha=0.2)

            # Testing cumulative cost
            te_mean = np.mean(r["test_matrix"], axis=0)
            te_std  = np.std(r["test_matrix"],  axis=0)
            ax_test.plot(test_ep, te_mean, color=color, linewidth=1.5, label=label)
            ax_test.fill_between(test_ep, te_mean - te_std, te_mean + te_std,
                                 color=color, alpha=0.2)

        # Avg steps bar chart
        step_means = [np.mean(r["avg_steps"]) for r in results]
        step_stds  = [np.std(r["avg_steps"])  for r in results]
        colors     = [r["color"] for r in results]
        ax_steps.bar(bar_x, step_means, bar_width, yerr=step_stds,
                     color=colors, capsize=5, alpha=0.8)
        ax_steps.set_xticks(bar_x)
        ax_steps.set_xticklabels([r["label"] for r in results], rotation=15, ha="right")
        ax_steps.set_title("Testing: Avg Steps per Episode")
        ax_steps.set_ylabel("Avg Steps")
        ax_steps.grid(True, axis="y", linestyle="--", alpha=0.5)

        for ax, title in [
            (ax_train, "Training: Cumulative Cost"),
            (ax_test,  "Testing: Cumulative Cost"),
        ]:
            ax.set_title(title)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Cumulative Cost")
            ax.legend(fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {OUT_FILE}")
        plt.show()

    # ================================================================== #
    # ORIGINAL MODE  (--agent or all agents)
    # ================================================================== #
    else:
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
