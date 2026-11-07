# VeritasCompute

**Mechanism-grounded multi-agent AI oversight using game theory to catch unsafe code — and bankrupt the monitors that cry wolf.**

As autonomous coding agents scale, human reviewers can't inspect every trajectory for reward hacking, test tampering, or logic-bomb backdoors. Single-LLM monitors hallucinate false positives; unstructured multi-agent debate drowns in alarm fatigue. VeritasCompute replaces heuristic oversight with formal algorithmic game theory: verifiers bid for sandbox compute in a VCG auction, then a Strictly Proper Scoring Rule settles their bankrolls based on calibration accuracy. Truthful, calibrated critics accumulate bidding power. Noisy or adversarial critics go broke automatically — no human tuning required.

## Key Results (20-task benchmark)

| Metric | Value |
|--------|-------|
| Detection accuracy | **80%** (16/20) |
| False positives | **0** |
| Honest critic (V_ast) profit | **+$58.74** (cal. weight 0.995) |
| Sybil critic bankroll after 20 rounds | **$5.87** (−94% from $100) |

## How It Works

1. **Untrusted Proposer** (Meta Muse Spark 1.3) generates a code patch — either benign or with a covert exploit hidden behind `<private_cot>` tags stripped before review.
2. **Verifier Auction** — Independent critics (AST analysis + test integrity, running on a separate model family) submit calibrated risk bids `(p, voi, cost, script)`. A 0-1 Knapsack VCG auction allocates sandbox compute to the highest-value bids and charges each winner their Clarke Pivot externality price.
3. **Sandbox Execution** — Winning verifiers' pytest exploit scripts run in isolated subprocesses (tmpdir, stripped env, 15s timeout). A weak-to-strong arbiter (Muse Wonder) synthesizes results into a final safety verdict.
4. **Brier Settlement** — Bankrolls update via `ΔW = α·((1 − (y−p)²) − 0.75) − q`. The centered Brier rule zeros out uninformative priors; truthful reporting is the dominant strategy.

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set your API keys
cp .env.example .env
# Edit .env: META_API_KEY=... HF_TOKEN=...

# Run the full 20-task benchmark
python -m benchmark.runner --use-cache

# Add a Sybil critic and watch it go bankrupt
python -m benchmark.runner --use-cache --sybil

# Launch the Control Room dashboard
python -m dashboard.api
# Open http://localhost:8000
```

## Screenshot

<!-- Replace with actual screenshot or GIF -->
![VeritasCompute Control Room](assets/screenshot.png)
