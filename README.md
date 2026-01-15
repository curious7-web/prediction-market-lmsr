# LMSR Trading Agents Under Belief Misspecification and Strategic Adversaries

This project implements a modular simulation framework for **binary prediction markets** based on the **Logarithmic Market Scoring Rule (LMSR)**. We use it to study how different trading agents — including **LLM-driven strategies** — behave under uncertainty, belief errors, and adversarial pressure.

The goal is not to “beat the market”, but to surface **failure modes**, **tail-risk exposure**, and **model brittleness** that emerge when agents make decisions with incomplete or incorrect beliefs.

---

## 🚀 Overview

Modern LLM agents can form beliefs about uncertain events (e.g., elections, sports, diagnostics). However, in markets:

- A slightly wrong belief can still be **profitable in expectation**…
- …but can produce **large negative tail outcomes** (rare but catastrophic).

This repo provides:

- A clean **LMSR simulation engine**
- Multiple **agent models** (truthful, noisy, LLM-based, robust, adversarial)
- Tools for **stress testing + tail risk analysis**
- Jupyter **experiment notebooks**
- Reproducible **metrics** (Mean, Worst, CVaR)

---

## 📦 Features

✔ **LMSR Market Maker**  
✔ **Agent Modules**:
- `TruthTeller` (trades toward true probability)
- `NoiseTrader` (random strategies)
- `GeminiStrategyAgent` (LLM belief or manual override)
- `RobustGeminiAgent` (risk-aware scaling)
- `AdversarialNoiseTrader` (strategic pressure)

✔ **Belief Misspecification Stress Tests**  
✔ **Tail Risk Metrics (CVaR)**  
✔ **Wealth Distribution Comparison**  
✔ **Adversary On/Off Ablations**  
✔ **Notebook Visualization**  

---

## 🛠 Installation

Clone and set up environment:

```bash
1. git clone https://github.com/curious7-web/prediction-market-lmsr.git
2. cd prediction-market-lmsr
3. python3 -m venv .venv
4. source .venv/bin/activate
5. pip install -r requirements.txt
7. Single Simulation
  ==> python -m src.simulate
8. Prints one episode outcome with fallback belief if Gemini API is unavailable:

[Gemini] API unavailable — using fallback belief.
9. Full Stress Test Demo
  ==> python -m src.demo
```
**Produces:**
1. Mean wealth

2. Worst outcomes

3. CVaR tails

4. Belief offset stress tests

5. Robust vs. Gemini comparison

6. Adversary on/off ablation

**Example snippet:**
```
=== Experiment 2: Adversary On vs Off ===
No adversary     | mean 0.06 | worst -0.39 | CVaR(5%) -0.33
With adversary   | mean 0.02 | worst -0.29 | CVaR(5%) -0.27
```
**📊 Notebook Experiments**

1. Launch Jupyter:

jupyter notebook notebooks/experiments.ipynb

2. Included visualizations:

a. Gemini vs RobustGemini wealth histograms

b. Adversary ON vs OFF distributions

c. CVaR cutoff markers

**Example insight:**

Tail-risk is interaction-dependent: adversarial traders can reduce catastrophic losses by disrupting directional exposure.
```
🧩 Project Structure
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
```
**📐 Metrics**

We evaluate agent wealth using:

a. Mean — profitability

b. Worst-case — drawdown severity

c. CVaR(5%) — tail conditional expectation

Mathematically:



This reveals risk that mean-based summaries hide.

**🧠 LLM Beliefs (Gemini)**

LLM beliefs are handled by:

**GeminiStrategyAgent(belief=0.55)**


If API access fails:

**[Gemini] API unavailable — using fallback belief.**


This ensures full reproducibility without paid API keys.

**⚠ Limitations**

This is a negative result–friendly research framework:

a. Agents do not update beliefs during trading

b. No multi-round learning or planning

c. Market liquidity is fixed (LMSR b)

d. LLM beliefs may be fallback 0.5 if API unavailable

e. These are intentional to isolate belief fragility & tail risk.

**📎 Relevance to Research & Workshops**

This framework surfaces limitations in:

a. LLM agents under uncertainty

b. Calibration vs. robustness

c. Risk sensitivity in interactive systems

d. Strategic multi-agent behavior

e. Suitable for workshops focused on:

f. LLM agents

g. Negative results & brittleness

h. Alignment & evaluation

i. AI safety & robustness

j. Mechanism design & markets

**🔁 Reproducibility Notes**

All experiments are seeded (numpy.random.seed)

No external data dependencies

Gemini API calls degrade gracefully

Single NVIDIA GPU not required

CVaR estimations use bootstrap-free MC

**📄 License**

MIT License (recommended for research)

**👤 Contact / Maintainer**

Author: Aditya Kumar Karna
For collaboration or research use, feel free to open an issue or PR.


---

### If you want, I can also generate:

✔ `LICENSE`  
✔ `setup.py` (pip installable)  
✔ Academic BibTex for citations  
✔ `paper.md` for JOSS journal  
✔ `abstract` for workshop submission  

Just say:

> **"Generate academic abstract"**

or

> **"Write ICBINB paper draft"**

or

> **"Make GitHub release notes"**

👑 Ready to push. Let me know if you want the README auto-committed.
