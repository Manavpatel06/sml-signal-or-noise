"""Stage D: can a node's failure be predicted from its voltage history?

For every node and every observed step BEFORE the node first enters
``real_anomaly_mask`` (still healthy), build features from the last W steps:
  voltage level, least-squares voltage slope, voltage residual after a per-node
  linear fit on temperature (voltage depends on temperature), and temperature.
Label = 1 if the node's first real anomaly happens within the next H steps.

Evaluation is leave-nodes-out cross-validation (nodes split into folds, so no
node is in both train and test). Classifier: Role 5's own LogisticRegression.
Baseline: the voltage level alone.

    python scripts/run_stage_d.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from signoise.config import RESULTS_DIR, TEMP, VOLT  # noqa: E402
from signoise.data import load_intel  # noqa: E402
from signoise.data.profile import real_anomaly_mask  # noqa: E402
from signoise.discrimination import LogisticRegression  # noqa: E402
from signoise.evaluation.metrics import roc_auc  # noqa: E402

FEATURES = ["volt_level", "volt_slope", "volt_residual", "temperature"]


def node_features(volt, temp, first_fail, window, horizons):
    """Rows for one node: (features, {H: label}) at healthy observed steps."""
    T = len(volt)
    ok = np.isfinite(volt) & np.isfinite(temp)
    fit = ok & (volt > 2.0) & (volt < 3.2) & (temp > -10) & (temp < 60)
    if fit.sum() < 10:
        return None
    b, a = np.polyfit(temp[fit], volt[fit], 1)
    end = first_fail if first_fail >= 0 else T
    feats, labels = [], {h: [] for h in horizons}
    t_idx = np.arange(window)
    for t in range(window, end):
        if not ok[t]:
            continue
        w = slice(t - window + 1, t + 1)
        vw, mw = volt[w], ok[w]
        if mw.sum() < 3:
            continue
        slope = np.polyfit(t_idx[mw], vw[mw], 1)[0]
        feats.append([volt[t], slope, volt[t] - (a + b * temp[t]), temp[t]])
        for h in horizons:
            labels[h].append(int(first_fail >= 0 and first_fail - t <= h))
    if not feats:
        return None
    return np.array(feats), {h: np.array(v) for h, v in labels.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=12, help="history length in steps")
    ap.add_argument("--horizons", type=int, nargs="+", default=[12, 72, 288],
                    help="steps ahead (bin 10: 12 ~ 1 h, 72 ~ 6 h, 288 ~ 1 day)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = load_intel()
    real = real_anomaly_mask(data)
    volt, temp = data.channel("voltage"), data.channel("temperature")
    first = np.array([np.argmax(real[:, n]) if real[:, n].any() else -1
                      for n in range(data.shape[1])])

    per_node = {}
    for n in range(data.shape[1]):
        r = node_features(volt[:, n], temp[:, n], first[n], args.window, args.horizons)
        if r is not None:
            per_node[n] = r
    nodes = np.array(sorted(per_node))
    rng = np.random.default_rng(args.seed)
    fold_of = dict(zip(nodes, rng.permutation(len(nodes)) % args.folds))
    print(f"{len(nodes)} nodes, {sum(len(v[0]) for v in per_node.values())} healthy steps; "
          f"{(first >= 0).sum()} nodes eventually fail")

    rows = []
    for h in args.horizons:
        ys, p_full, p_level = [], [], []
        for f in range(args.folds):
            trn = [n for n in nodes if fold_of[n] != f]
            tst = [n for n in nodes if fold_of[n] == f]
            Xtr = np.vstack([per_node[n][0] for n in trn])
            ytr = np.concatenate([per_node[n][1][h] for n in trn])
            Xte = np.vstack([per_node[n][0] for n in tst])
            yte = np.concatenate([per_node[n][1][h] for n in tst])
            if len(np.unique(ytr)) < 2:
                continue
            full = LogisticRegression(l2=1e-3, lr=0.5, max_iter=3000).fit(Xtr, ytr)
            p_full.append(full.predict_proba(Xte)[:, 1])
            p_level.append(-Xte[:, 0])  # lower voltage -> higher risk
            ys.append(yte)
        y = np.concatenate(ys)
        pf, pl = np.concatenate(p_full), np.concatenate(p_level)
        rows.append({"horizon_steps": h, "horizon_hours": round(h * 31 * 10 / 3600, 1),
                     "positive_rate": round(float(y.mean()), 4),
                     "auc_logreg": roc_auc(y, pf), "auc_voltage_only": roc_auc(y, pl)})
        print(f"H={h:>4} steps (~{rows[-1]['horizon_hours']} h): "
              f"AUC logreg={rows[-1]['auc_logreg']:.3f}  "
              f"voltage only={rows[-1]['auc_voltage_only']:.3f}  "
              f"positives={y.mean():.1%}")

    out = RESULTS_DIR / "stage_d"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "stage_d_auc.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(df["horizon_hours"], df["auc_logreg"], "o-", label="logistic regression (4 features)")
    ax.plot(df["horizon_hours"], df["auc_voltage_only"], "s--", label="voltage level only")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("prediction horizon (hours, log scale)")
    ax.set_ylabel("AUC (leave-nodes-out CV)")
    ax.set_title("Predicting failure before bad readings appear")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "stage_d_auc.png", dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
