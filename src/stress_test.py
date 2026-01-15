from typing import Dict, List
import numpy as np

from simulate import run_episode


def run_stress_test(
    n_runs: int = 200,
    b: float = 20.0,
    T: int = 100,
    true_probs: List[float] = None,
) -> Dict:
    """
    Run multiple simulations with different random seeds
    and aggregate EV + regret statistics.
    """
    if true_probs is None:
        true_probs = [0.55, 0.45]

    wealth_records = {"truth": [], "noise": []}
    regret_records = {"truth": [], "noise": []}
    final_price_records = []

    for seed in range(n_runs):
        result = run_episode(
            K=len(true_probs),
            b=b,
            T=T,
            true_probs=true_probs,
            seed=seed,
        )

        for agent, wealth in result["final_wealth"].items():
            wealth_records[agent].append(wealth)

        for agent, regret in result["regret"].items():
            regret_records[agent].append(regret)

        final_price_records.append(result["final_prices"])

    summary = {}

    for agent in wealth_records:
        wealth_arr = np.array(wealth_records[agent])
        regret_arr = np.array(regret_records[agent])

        summary[agent] = {
            "mean_wealth": float(np.mean(wealth_arr)),
            "std_wealth": float(np.std(wealth_arr)),
            "min_wealth": float(np.min(wealth_arr)),
            "max_wealth": float(np.max(wealth_arr)),
            "mean_regret": float(np.mean(regret_arr)),
            "worst_regret": float(np.max(regret_arr)),
        }

    summary["market"] = {
        "mean_final_price": list(np.mean(final_price_records, axis=0)),
        "std_final_price": list(np.std(final_price_records, axis=0)),
        "n_runs": n_runs,
        "liquidity_b": b,
        "timesteps": T,
    }

    return summary


if __name__ == "__main__":
    summary = run_stress_test(n_runs=200)

    print("\n=== STRESS TEST RESULTS ===\n")

    for agent, stats in summary.items():
        if agent == "market":
            continue

        print(f"Agent: {agent}")
        print(f"  Mean wealth: {stats['mean_wealth']:.3f}")
        print(f"  Std wealth: {stats['std_wealth']:.3f}")
        print(f"  Min wealth: {stats['min_wealth']:.3f}")
        print(f"  Max wealth: {stats['max_wealth']:.3f}")
        print(f"  Mean regret: {stats['mean_regret']:.3f}")
        print(f"  Worst regret: {stats['worst_regret']:.3f}")
        print()

    print("Market behavior:")
    print("  Mean final price:", summary["market"]["mean_final_price"])
    print("  Std final price:", summary["market"]["std_final_price"])
    print("  Runs:", summary["market"]["n_runs"])
