# Role 6 — Evaluation & results (Ziyad)

Files: `src/signoise/injection/inject.py` (v0 works), new
`scripts/experiments/*.py`, the results table and every figure in the talk.

## Tasks

1. **Better injection** (keep the `inject_anomalies` signature; add options):
   - `quiet_only=True`: choose injection windows where the rolling variance
     across nodes is low and the node was observed (use the `valid` mask idea
     from the harness).
   - Harder variants as optional kinds: stuck-at value, slow drift, event with
     a lag of 1-3 steps between neighbours.
2. **Graded difficulty:** sweep `magnitude` in {1, 2, 3, 4, 6} and plot AUC
   (overall, events, failures) vs magnitude for every implemented detector.
3. **Statistics:** repeated temporal CV (several seeds), mean ± std and 95%
   confidence intervals; a paired comparison of each detector vs k-sigma.
4. **Ablation (with Tejas/Manav):** drop one Stage C feature at a time and
   report the change in event-vs-failure accuracy; include the random-neighbour
   control.
5. **Results table + figures** in `results/` and a consolidated
   `results/RESULTS.md` the slides and report use. Own figure style
   (matplotlib, consistent colours per detector, axis labels with units).
6. Tests for any new injection kind: labels match exactly the cells changed.

Done when one command (`python scripts/experiments/run_all.py`) reproduces
every number and figure in the report.
