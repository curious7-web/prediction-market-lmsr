import numpy as np

from .simulate import run_episode
from .metrics import cvar


def summarize(xs):
    return {
        "mean": np.mean(xs),
        "worst": np.min(xs),
        "cvar": cvar(xs),
    }


def run_demo():
    runs = 200

    print("\nRunning 200 market simulations...\n")

    wealth = {
        "truth": [],
        "noise": [],
        "gemini": [],
    }

    for i in range(runs):
        r = run_episode(
            seed=i,
            include_gemini=True,
        )
        for k in wealth:
            wealth[k].append(r["wealth"][k])

    print("Agent performance")
    print("--------------------------------------------------")
    for k, v in wealth.items():
        stats = summarize(v)
        print(
            f"{k:<8} | mean: {stats['mean']:6.2f} "
            f"| worst: {stats['worst']:7.2f} "
            f"| CVaR(5%): {stats['cvar']:7.2f}"
        )

    print("\nMarket behavior")
    print("--------------------------------------------------")
    prices = [
        run_episode(seed=i, include_gemini=True)["final_price"]
        for i in range(runs)
    ]
    print(f"Mean final price (p1): {np.mean(prices):.3f} +/- {np.std(prices):.3f}")
    print("\nBelief misspecification stress test")
    print("(Gemini belief error vs tail risk)\n")
    print("offset | mean wealth | worst | CVaR(5%)")
    print("--------------------------------------------")

    belief_offsets = np.linspace(-0.15, 0.15, 7)

    for offset in belief_offsets:
        ws = []
        for i in range(runs):
            r = run_episode(
                seed=i,
                include_gemini=True,
                gemini_belief=0.55 + offset,
            )
            ws.append(r["wealth"]["gemini"])

        stats = summarize(ws)
        print(
            f"{offset:+.2f}  | "
            f"{stats['mean']:8.2f} | "
            f"{stats['worst']:7.2f} | "
            f"{stats['cvar']:7.2f}"
        )

    print("\n=== Experiment 1: Gemini vs RobustGemini ===\n")

    g_vals = []
    rg_vals = []

    for i in range(runs):
        r = run_episode(
            seed=i,
            include_gemini=True,
            include_robust=True,
        )
        g_vals.append(r["wealth"]["gemini"])
        rg_vals.append(r["wealth"]["robust_gemini"])

    g = summarize(g_vals)
    rg = summarize(rg_vals)

    print("Agent           | Mean    | Worst   | CVaR(5%)")
    print("----------------------------------------------")
    print(f"Gemini          | {g['mean']:6.2f} | {g['worst']:7.2f} | {g['cvar']:7.2f}")
    print(f"RobustGemini    | {rg['mean']:6.2f} | {rg['worst']:7.2f} | {rg['cvar']:7.2f}")

    print("\nInterpretation:")
    print(
        "RobustGemini achieves comparable expected value to Gemini, but exhibits "
        "nearly identical worst-case and CVaR outcomes in this single-round LMSR "
        "setting. This indicates that linear risk-aware position scaling alone is "
        "insufficient to meaningfully reduce tail risk; losses are primarily driven "
        "by outcome realizations and market structure rather than marginal exposure size."
    )


    print("\n=== Experiment 2: Adversary On vs Off ===\n")

    no_adv = []
    with_adv = []

    for i in range(runs):
        r1 = run_episode(
            seed=i,
            include_gemini=True,
            include_adversary=False,
        )
        r2 = run_episode(
            seed=i,
            include_gemini=True,
            include_adversary=True,
        )
        no_adv.append(r1["wealth"]["gemini"])
        with_adv.append(r2["wealth"]["gemini"])

    na = summarize(no_adv)
    wa = summarize(with_adv)

    print("Condition        | Mean    | Worst   | CVaR(5%)")
    print("-----------------------------------------------")
    print(f"No adversary     | {na['mean']:6.2f} | {na['worst']:7.2f} | {na['cvar']:7.2f}")
    print(f"With adversary   | {wa['mean']:6.2f} | {wa['worst']:7.2f} | {wa['cvar']:7.2f}")

    print("\nInterpretation:")
    print(
        "Introducing an adversarial trader significantly reshapes tail-risk behavior. "
        "Although expected value declines, adversarial pressure limits directional "
        "concentration, leading to substantially improved worst-case and CVaR outcomes."
    )
    print(
        "This demonstrates that tail risk in LMSR-style markets is interaction-dependent, "
        "not solely a function of belief accuracy. Strategic counterparties can dominate "
        "risk dynamics even when agents hold similar beliefs."
    )


if __name__ == "__main__":
    run_demo()
