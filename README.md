# Beyond the Read

**Uncertainty-aware graph-temporal reconstruction of hidden physical states from imperfect RFID streams.**

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USER/beyond-the-read-rfid/blob/main/notebooks/Beyond_the_Read_RFID_AI_Colab_v2_1.ipynb)

Reproducibility repository for the manuscript submitted to *Pervasive and Mobile Computing*
(Elsevier). Everything reported in the paper — every table, figure and statistic — is regenerated
by the notebook in this repository from a single configuration, and every artefact is stamped with
the hash of the configuration that produced it.

---

## What this is about

An RFID read is not the same thing as a physical event. Reads go missing, duplicate, arrive late,
carry noisy RSSI, or appear at readers the asset never passed. Conventional RFID analytics responds
by classifying each read as normal or anomalous. We argue that is the wrong question. The
operationally meaningful quantity is the **latent trajectory** of the asset, of which the reads are
only unreliable evidence.

This repository implements that reformulation and tests it honestly:

- **A graph-constrained neural CRF (GT-CRF)** that places admissible process transitions inside the
  training objective, not in a post-hoc filter. Transitions are parameterised as
  `A = B_G + δ·tanh(W)` with bounded `δ`, so gradient descent cannot quietly cancel the process
  prior — a failure mode we document.
- **Dual anomaly diagnosis** separating *sensing* anomalies (valid process, corrupted evidence) from
  *operational* deviations (genuinely wrong physical route), with label-free mechanistic scores
  alongside supervised references.
- **Degradation-adaptive conformal prediction.** Marginal conformal coverage silently collapses as
  sensing degrades; stratifying calibration by a severity estimate computed from the observed stream
  alone restores it, with no access to ground truth at deployment.
- **Exact minimum-cost counterfactual repair** — the true global optimum, solved as a shortest path
  to the language of admissible routes, not a heuristic gap-filler.
- **An audited leakage firewall.** Sixteen observable variables are permitted to reach a model; ten
  oracle variables are quarantined for evaluation only. The audit runs at every model boundary and
  raises an error if the firewall is breached.

---

## Headline results

Reported for the `QUICK_RUN` configuration (3000 assets, 3 seeds, config hash `e37c5b50709b`).

| Finding | Evidence |
|---|---|
| **Read-level evaluation inverts the ranking of methods.** The model with the best per-read accuracy (0.980) produces inadmissible transitions on 14.9% of decoded steps. GT-CRF is *last of seven* on per-read accuracy (0.934) yet best on process consistency and second on route quality. | `Table_03`, `Fig_02` |
| **Structured reconstruction clearly beats unstructured.** Trajectory similarity 0.901 vs 0.820 for the raw stream; inadmissible transitions cut 8.6×; Cliff's δ = 0.31, Holm-adjusted *p* < 0.0001. | `Table_03`, `Table_22` |
| **Marginal conformal prediction fails silently under degradation.** Coverage falls 0.95 → 0.35 as severity rises, while the fixed quantile gives no indication anything changed. Severity-conditional (Mondrian) calibration holds 0.86–0.94 using only observable quantities. | `Table_06`, `Fig_04` |
| **Exact repair is cheap and sparse.** Process consistency restored on 100% of trajectories at 0.50 edits and 0.113 ms each — 7.4× faster than inference itself. | `Table_07`, `Table_23` |
| **Label-free diagnosis works for sensing, not for operations.** Observation distortion reaches AUROC 0.850 / AUPRC 0.931 for `A_s` with no labels; mechanistic scores for `A_o` sit near chance (0.578–0.605). | `Table_09` |

### Results that did not support our hypotheses

We report these because they are findings, not because we have to.

- **End-to-end graph integration did not beat post-hoc constrained decoding.** RF + Viterbi reaches
  trajectory similarity 0.9077 against 0.9006 for GT-CRF, and is consistently higher across seeds.
  Holm-adjusted *p* = 0.227, Cliff's δ = 0.002 (negligible). The end-to-end model's defensible
  advantages are its process consistency (0.027 vs 0.039) and the coherent posterior the calibration
  and conformal layers depend on — not reconstruction accuracy.
- **No abstention threshold could be certified.** Learn-then-Test found no threshold meeting a 10%
  selective-risk target, because exact-route error never falls below 0.238 at *any* coverage. The
  resulting conservative fallback made the decision policy the **lowest-utility** of four evaluated
  under most cost assumptions. A risk–coverage curve always looks reassuring; a procedure with a
  finite-sample guarantee is what reveals there is no threshold at which the promise holds.

**Scale caveat.** These come from a reduced configuration (3000 assets, 3 seeds, 12 epochs). The
negative reconstruction result rests on a margin of 0.004 and should be read as *not demonstrated*
rather than refuted. A full run (12000 assets, 10 seeds, 40 epochs) is the intended replication.

---

## Repository layout

```
.
├── notebooks/
│   └── Beyond_the_Read_RFID_AI_Colab_v2_1.ipynb   # the complete pipeline, 52 cells
├── results/
│   ├── tables/                                    # 45 result tables (CSV)
│   ├── latex/                                     # 27 booktabs LaTeX fragments
│   ├── figures/                                   # 23 figures (PNG 300 dpi + PDF)
│   ├── config.json                                # full configuration + environment capture
│   └── Outputs_Summary.txt                        # human-readable run summary
├── paper/
│   ├── main.tex, Sections/, Figures/              # manuscript source
│   └── main.pdf
├── figures_src/                                   # scripts regenerating Figs. 1-4
│   ├── diagram_lib.py
│   ├── fig_framework.py
│   ├── fig_operational.py
│   ├── fig_training.py
│   └── fig_reconstruction.py
├── requirements.txt
├── LICENSE                                        # MIT (code)
├── LICENSE-DATA                                   # CC BY 4.0 (results, figures)
└── CITATION.cff
```

---

## Reproducing the results

### Option 1 — Colab (no setup)

Open the notebook with the badge above and run all cells. A GPU runtime is recommended but not
required; the `QUICK_RUN` configuration completes in a few minutes on CPU.

### Option 2 — Local

```bash
git clone https://github.com/USER/beyond-the-read-rfid.git
cd beyond-the-read-rfid
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/Beyond_the_Read_RFID_AI_Colab_v2_1.ipynb
```

The notebook mounts Google Drive by default (cell 1). Running locally, replace `BASE_DIR` with a
local path.

### Configuration

All experimental settings live in a single `CFG` dictionary in cell 3:

```python
QUICK_RUN = True    # set False for the full manuscript run

CFG = dict(
    n_assets      = 3000 if QUICK_RUN else 12000,
    n_seeds       = 3    if QUICK_RUN else 10,
    epochs        = 12   if QUICK_RUN else 40,
    stress_levels = 6    if QUICK_RUN else 9,
    lambda_graph  = 8.0,     # penalty on inadmissible transitions
    trans_delta   = 0.25,    # bound on learned modulation of the graph prior
    alpha         = 0.10,    # conformal miscoverage target
    risk_epsilon  = 0.10,    # Learn-then-Test selective-risk budget
    ...
)
```

The configuration is hashed to 12 hex characters and that hash is written into **every** exported
table and figure. If a number in the paper and a number in `results/` disagree, the hash tells you
which run each came from.

**Expected runtime**, `QUICK_RUN=True`, single T4: ~12 min end to end. GT-CRF training is 25 s of
that; the bulk is the stress sweeps, which retrain nothing but simulate many corpora.

---

## Notes for anyone building on this

**The leakage firewall is the part we would most encourage you to copy.** Simulation-based RFID
studies routinely leak corruption flags into feature sets, which inflates every number downstream
and is nearly invisible in a paper. `assert_no_leakage()` is called at the sequence tensorizer, the
baseline feature matrices, the severity estimator and the diagnostic feature set, and it fails
loudly. It costs almost nothing and it is the difference between a result and an artefact.

**Bound your learned transition potentials.** With an unbounded learned transition term, gradient
descent drives `W` to cancel `B_G`; the process prior disappears from the objective and you are left
with a conventional neural CRF wearing the vocabulary of process knowledge. The `δ·tanh(W)`
parameterisation prevents this.

**Do not select on per-read accuracy** if you care about trajectories. Validation route similarity
peaks at epoch 0 and *declines* while per-read accuracy rises monotonically. The two objectives are
close to antagonistic here.

---

## Citation

```bibtex
@article{zayoud2026beyondtheread,
  author  = {Zayoud, Rahma and Hamam, Habib},
  title   = {Beyond the Read: Uncertainty-Aware Graph-Temporal Intelligence for
             Reconstructing Hidden Physical States from Imperfect {RFID} Streams},
  journal = {Pervasive and Mobile Computing},
  note    = {Under review},
  year    = {2026}
}
```

---

## Licence

Code is released under the **MIT Licence**. Result tables and figures are released under
**CC BY 4.0**. If you use the simulator or the evaluation protocol, a citation is appreciated.

## Contact

Rahma Zayoud — <zayoud.rahma@gmail.com> (corresponding author)
Habib Hamam — <habib.hamam@umoncton.ca>

Faculty of Engineering, Université de Moncton, Canada.
