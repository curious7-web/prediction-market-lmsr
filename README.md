# LMSR Trading Agents Under Belief Misspecification and Strategic Adversaries

This project implements a modular simulation framework for **binary prediction markets** based on the **Logarithmic Market Scoring Rule (LMSR)**. We use it to study how different trading agents — including **LLM-driven strategies** — behave under uncertainty, belief errors, and adversarial pressure.

The goal is not to “beat the market”, but to surface **failure modes**, **tail-risk exposure**, and **model brittleness** that emerge when agents make decisions with incomplete or incorrect beliefs.

---

## 🚀 Overview

Modern LLM agents can form beliefs about uncertain events (e.g., elections, sports, diagnostics). However, in markets:

- A slightly wrong belief can still be **profitable in expectation**…
- …but can produce **rare but catastrophic tail outcomes**.

This repo provides:

- A clean **LMSR simulation engine**
- Multiple **agent models** (truthful, noisy, LLM-based, robust, adversarial)
- Tools for **stress testing + tail risk analysis**
- Reproducible **metrics** (Mean, Worst, CVaR)
- Jupyter **experiment notebooks**

---

## 📦 Features

✔ LMSR market maker  
✔ Agent modules:
- `TruthTeller` (trades toward true probability)
- `NoiseTrader` (random trading)
- `GeminiStrategyAgent` (LLM belief or manual override)
- `RobustGeminiAgent` (risk-aware scaling)
- `AdversarialNoiseTrader` (strategic pressure)

✔ Belief misspecification stress tests  
✔ Tail risk metrics (CVaR)  
✔ Wealth distribution comparison  
✔ Adversary on/off ablations  
✔ Notebook visualization  

---

## 🛠 Installation

Clone and set up environment:

    git clone https://github.com/curious7-web/prediction-market-lmsr.git
    cd prediction-market-lmsr
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

---

## ▶️ Running Simulations

### Single simulation

    python -m src.simulate

If Gemini API is unavailable:

    [Gemini] API unavailable — using fallback belief.

### Full stress test

    python -m src.demo

This produces:

- Mean wealth
- Worst outcomes
- CVaR tails
- Belief offset stress tests
- Gemini vs RobustGemini comparison
- Adversary on/off ablations

Example output:

    === Experiment 2: Adversary On vs Off ===
    No adversary     | mean 0.06 | worst -0.39 | CVaR(5%) -0.33
    With adversary   | mean 0.02 | worst -0.29 | CVaR(5%) -0.27

---

## 📊 Notebook Experiments

Launch:

    jupyter notebook notebooks/experiments.ipynb

Visualizations include:

- Gemini vs RobustGemini wealth histograms
- CVaR tail markers
- Adversary vs no-adversary distributions

Key insight:

> Tail risk is interaction-dependent: adversarial traders can reduce catastrophic losses by disrupting directional exposure.

---

## 🧩 Project Structure

    prediction-market-lmsr/
     ├── src/
     │    ├── simulate.py         # single-episode simulator
     │    ├── demo.py             # batch experiments + stress tests
     │    ├── agents.py           # agent definitions
     │    ├── lmsr.py             # LMSR market maker
     │    ├── metrics.py          # CVaR & evaluation metrics
     │    └── gemini_interface.py # Gemini API (fallback safe)
     ├── notebooks/
     │    └── experiments.ipynb   # visualization + analysis
     ├── requirements.txt
     ├── README.md
     └── LICENSE (optional)

---

## 📐 Metrics

We evaluate agent wealth under three risk dimensions:

1. **Mean** — profitability  
2. **Worst-case** — drawdown severity  
3. **CVaR(5%)** — tail conditional expectation

In symbols:

\[
\text{CVaR}_\alpha = \mathbb{E}[X \mid X \leq \text{VaR}_\alpha]
\]

This reveals tail risk that mean-based summaries can hide.

---

## 🧠 LLM Beliefs (Gemini)

LLM beliefs are handled via:

- `GeminiStrategyAgent(belief=0.55)` for manual belief override
- Or via `GeminiClient.propose_belief(market_context)` when configured

If API access fails:

    [Gemini] API unavailable — using fallback belief.

This ensures **reproducibility without paid API keys**.

---

## ⚠ Limitations

This is a **negative-result–friendly** research framework:

- Agents do **not** update beliefs during trading
- No multi-round planning or learning
- Market liquidity parameter `b` is fixed
- LLM fallback belief can be constant (0.5)
- API usage is optional and degrades gracefully

These simplify analysis of **belief fragility** and **tail risk**.

---

## 📎 Relevance

This framework reveals limitations in:

- LLM calibration and confidence
- Risk sensitivity under misspecification
- Multi-agent interaction effects
- Robustness vs pure expectation maximization
- Mechanism design & financial AI with LLM agents

Suitable for workshops on:

- LLM agents
- Alignment & evaluation
- Negative results (ICBINB)
- Robust decision-making
- Financial AI / mechanism design

---

## 🔁 Reproducibility Notes

- All experiments seeded (`numpy.random.seed`)
- No external data dependencies
- No GPU required
- Gemini API calls fail safely with explicit logging
- CVaR computed via Monte Carlo over repeated runs

---

## 📄 License

MIT License (recommended for research).

---

## 👤 Contact / Maintainer

**Author:** Aditya Kumar Karna  

Open issues or pull requests are welcome.
