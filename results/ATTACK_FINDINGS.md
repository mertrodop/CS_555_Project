# Shilling Attack Findings — LLM-AGR Robustness Study

## Setup
- Dataset: see attack_grid.csv
- Attack strategy: bandwagon
- Variants: full, wo_ags_ib, wo_kd, wo_se, base
- Attack sizes: 0%, 5%, 10%, 15%, 25% of genuine users
- 3 random seeds per condition; reported as mean ± std

## Recall@20 Degradation (Δ from clean to 25% attack)

  - LLM-AGR (full)      : Δrecall@20 = -0.0105
  - BIGCF (base)        : Δrecall@20 = -0.0135

## Key Observations

### Which component absorbs the attack?
[TO BE FILLED after results]

Compare slopes: a smaller absolute Δrecall means better robustness.
- **full vs wo_ags_ib**: Does removing Adaptive Graph Structure + IB cause larger degradation?
  If yes, AGS+IB acts as a defence mechanism (denoises attacked edges).
- **full vs wo_kd**: If KD removal hurts robustness, the knowledge distillation path
  (preference alignment) provides some regularisation that suppresses attacker influence.
- **full vs wo_se**: If removing semantic embeddings causes larger drop, LLM embeddings
  provide an out-of-band signal that is harder for attackers to spoof.
- **base (BIGCF) vs full**: Quantifies the net robustness gain from LLM-AGR as a whole.

### Attack success (target_hr@20)
[TO BE FILLED after results]

Compare target_hr@20 across variants: a higher hit ratio means the attack was more
effective at promoting target items. Variants that suppress target_hr better are more
robust from an adversarial perspective.

## Plots
- `robustness.png`      — Recall@20 vs attack size (quality degradation)
- `attack_success.png`  — target_hr@20 vs attack size (attacker perspective)
