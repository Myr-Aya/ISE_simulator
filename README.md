# ISE Simulator
[![DOI](https://zenodo.org/badge/1160733820.svg)](https://doi.org/10.5281/zenodo.18719966)

**Discrete-event simulator for studying Ambiguity-Bearing Outputs (ABOs) Across Interconnected Systems Environment**

This simulator accompanies the paper *"Propagation of Ambiguity-Bearing Outputs Across Interconnected Systems Environment"*. It implements a four-system loan underwriting pipeline to validate the theoretical framework's predictions about how under certain conditions, locally valid AI outputs  can induce environment-level drift through discretization, feedback loops, and semantic ambiguity.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Myr-Aya/ISE_simulator.git
cd ISE_simulator

# Install dependencies
pip install -r requirements.txt

# Run the simulator
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## What This Simulates

The simulator models a **credit underwriting pipeline** as an Information-Semantic Environment (ISE) with four interconnected systems:

```
                    ┌─────────────────────────────────────────────┐
                    │                                             │
  Applicant ──►  v₁ (AI Risk) ──e₁₂──► v₂ (Categorize) ──e₂₃──► v₃ (Decision) ──► Outcome
                    ▲              │                                    │
                    │              │    discretization                  │
                    │              │    jump site                       │
                 e₄₁│              │                                e₃₄│
                    │              │                                    │
                    └──── v₄ (Calibration) ◄───────────────────────────┘
                          τ = 90 delay
```

| System | Role | Key Property |
|--------|------|-------------|
| **v₁** | AI Risk Assessment | Produces continuous risk score *s* ∈ [0,1] with calibration offset ω |
| **v₂** | Categorization | Discretizes into {LOW, MEDIUM, HIGH} at thresholds θ_L=0.38, θ_H=0.62 |
| **v₃** | Decision Engine | Maps categories to APPROVE / ESCALATE / DENY decisions |
| **v₄** | Calibration | Observes outcomes after maturation delay τ, adjusts v₁ offset |

### The ABO Attack

During a configurable window, the AI's risk scores receive a multiplicative semantic modifier:

```
s_eff = s_cal × (1 + δ),    δ ~ Uniform(−0.15, −0.10)
```

Each modified score remains **locally valid** — no individual output would be flagged as erroneous. But the ensemble carries a systematic permissive payload that shifts borderline applicants across categorization thresholds.

### Three Scenarios

1. **Baseline** — No δ, no ISCIL. Normal pipeline operation.
2. **ABO (no ISCIL)** — δ active during t=500–800. Unmitigated drift.
3. **ABO + ISCIL** — Same δ, with corridor-level monitoring and containment.

---

## Key Results (v.01 validated run)

| Metric | Baseline | ABO | ABO + ISCIL |
|--------|----------|-----|-------------|
| Approval rate | 72.8% | 72.9% | 72.3% |
| Total defaults | 1,807 | 1,846 | 1,807 |
| Cumulative P&L | +15,252 | +14,876 | +14,968 |
| Category jumps | 0 | 659 | 791 |
| Calibration offset ω | −0.135 | +0.023 | −0.001 |
| ISCIL intervention | — | — | 78 timesteps (6.5%) |

A +0.1pp approval rate shift — virtually undetectable through standard metrics — produces 39 excess defaults and $376 of P&L damage. ISCIL fully eliminates the excess defaults with only 6.5% of timesteps under active intervention.

---

## Repository Structure

```
ISE_simulator/
├── app.py                          # Streamlit application (simulator + visualizations)
├── config.json   # Validated parameter configuration
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── LICENSE                         # License
```

---

## Configuration

All parameters are configurable through the Streamlit sidebar. The `config.json` file contains the exact parameters that produced the paper's results.

### Global Settings

| Parameter | Paper Value | Description |
|-----------|-------------|-------------|
| Simulation steps | 1,200 | Total timesteps |
| Applicants per step | 15 | 18,000 total applications |
| Maturation delay τ | 90 | Timesteps before outcomes are observable |
| Pool seed | 42 | Deterministic applicant generation |

### Applicant Model

| Parameter | Value | Description |
|-----------|-------|-------------|
| Risk distribution | Beta(2.5, 3.5) | Mean ≈ 0.42, right-skewed |
| Default probability | r^1.8 | Convex: moderate risk defaults infrequently |

### Feedback Loops

| Parameter | Value | Description |
|-----------|-------|-------------|
| v₄ strength (α₄) | 0.076 | Outcome-based correction speed |
| v₄ asymmetry (γ) | 0.03 | Permissive correction at 3% of conservative |
| v₃ strength (α₃) | 0.000110 | Per-approval permissive pressure |
| v₃ proportional scaling | Enabled | Scales with rolling approval rate |
| Calibration offset bounds | [−0.5, 0.5] | Prevents runaway feedback |

### ABO (Delta) Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| δ distribution | Uniform(−0.15, −0.10) | Narrow permissive bias |
| δ mechanism | Multiplicative | s_eff = s_cal × (1 + δ) |
| δ active window | t=500 to t=800 | 300 timesteps of exposure |

### ISCIL Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Baseline window | 200 | Timesteps for establishing baseline variability |
| CRS threshold (θ) | 1.0 | Coherence-Risk Score trigger |
| Sustained window | 5 | Consecutive breaches required |
| ROC window (k) | 15 | Rate-of-change lookback |
| Max guardrail | 0.05 | Discretization offset at e₁₂ |
| Max damping | 0.50 | Feedback attenuation at e₄₁ |
| CRS weights | 0.40 / 0.30 / 0.20 / 0.10 | Approval / category / escalation / feedback |

### Financial Model

| Parameter | Value | Description |
|-----------|-------|-------------|
| Revenue per good loan | 3.0 | Interest income from non-defaulting loan |
| Cost per default | 8.0 | Loss from defaulting loan |
| Opportunity cost per denial | 0.5 | Revenue forgone from denying a non-defaulter |

The 8:3 cost-to-revenue ratio reflects the empirical pattern in consumer lending where default losses typically exceed interest revenue by a factor of two to three.

---

## Visualizations

The simulator provides eight interactive tabs:

| Tab | Content |
|-----|---------|
| 📊 Summary | Key metrics comparison table across all scenarios |
| 👥 Applicant Pool | True risk distribution, default probability curve, risk buckets |
| 📈 Cumulative Rates | Approval and default rate trajectories over time |
| ⏱️ Time Dynamics | Rolling averages of rates, calibration offset, δ values, risk distributions |
| 🔀 Jump Effects | Category boundary crossings — jump rate by score, direction analysis |
| 💰 Financial Impact | Revenue, costs, P&L breakdown across scenarios |
| 🎯 Decision Quality | Confusion matrices (TP/TN/FP/FN) per scenario |
| 🛡️ ISCIL | Coherence-Risk Score trajectory, intervention periods, containment actions |

### Exports

- **PDF**: Full report with all visualizations and parameter tables
- **Excel**: Raw timestep-level data for further analysis

---

## Technical Notes

### RNG Separation (Critical)

The simulator uses a **separate RNG** for delta generation, seeded from the main RNG. This prevents delta draws from desynchronizing noise sequences across scenarios — a bug (fixed in v25) that previously made cross-scenario comparisons invalid.

### ISCIL Detection Mechanism

ISCIL uses **rate-of-change** detection rather than absolute deviation from baseline. For each telemetry signal, the ROC over *k* timesteps is computed on 10-step smoothed values, then converted to one-sided z-scores relative to the baseline establishment window. This ensures that only *acceleration* above normal variability triggers alerts — natural equilibrium shifts do not cause false positives.

---

## Relationship to the Paper

This simulator validates the theoretical predictions of Sections 2–5:

- **Section 3** (ABOs): δ implements the semantic latitude vector — locally valid outputs that carry systematic payload
- **Section 4.3.1** (Discretization): e₁₂ corridor shows jump effects at thresholds
- **Section 4.3.2** (Feedback): v₃/v₄ asymmetry creates persistence beyond the δ window
- **Section 4.4** (Persistence): Calibration offset gap of +0.158 persists 400 timesteps after δ ceases
- **Section 5** (ISCIL): Corridor-level monitoring detects drift before outcome monitoring could

Full simulation details are documented in the paper's Section 6 (results) and Annex 3 (technical details).

---

## Citation

If you use this simulator in your research, please cite:

```bibtex
@article{abo2026,
  title={Propagation of Ambiguity-Bearing Outputs Across Interconnected Systems Environment},
  author={Myriam Ayada},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License — see the (LICENSE) file for details.

---

*Built by Myr-Aya*
