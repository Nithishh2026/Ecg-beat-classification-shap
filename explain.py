"""SHAP analysis for the trained CNN (GradientExplainer) and the RF baseline (TreeSHAP).

    python explain.py --out results

Produces figures plus two numeric checks so the explanations are not just pretty pictures:
  * region attribution: share of |SHAP| falling on P/PR, QRS and ST/T per class
    (plus the share on the first/last 5 samples, an edge-artifact detector)
  * deletion test: masking the top-k SHAP samples should hurt the predicted-class
    probability much more than masking k random samples.
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import shap
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import FEATURE_NAMES, FS, HALF, REGIONS, SHORT, extract_features  # noqa: E402
from model import BeatCNN  # noqa: E402

CLASS_COLORS = ["#0072B2", "#D55E00", "#009E73"]  # N, S, V (colorblind-safe)
INK, MUTED = "#111827", "#6b7280"


def to_array(sv, n_classes=3):
    """shap returns a list of per-class arrays (old) or one array with classes last (new)."""
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    sv = np.asarray(sv)
    assert sv.shape[-1] == n_classes, sv.shape
    return sv


@torch.no_grad()
def proba(model, X):
    return model(torch.as_tensor(X, dtype=torch.float32).unsqueeze(1)).softmax(1).numpy()


def balanced_background(X, y, n_per_class, rng):
    idx = []
    for c in range(3):
        pool = np.where(y == c)[0]
        idx += list(rng.choice(pool, min(n_per_class, len(pool)), replace=False))
    return X[np.array(idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--k-mask", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    d = np.load(out / "arrays.npz")
    Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    model = BeatCNN()
    model.load_state_dict(torch.load(out / "cnn.pt"))
    model.eval()

    probs_te = proba(model, Xte)
    pred_te = probs_te.argmax(1)

    # ---- choose which test beats to explain: per-class sample + some misclassified ------- #
    chosen = []
    for c in range(3):
        pool = np.where(yte == c)[0]
        chosen += list(rng.choice(pool, min(args.per_class, len(pool)), replace=False))
    wrong = np.where(pred_te != yte)[0]
    chosen += list(rng.choice(wrong, min(9, len(wrong)), replace=False)) if len(wrong) else []
    chosen = np.array(sorted(set(chosen)))
    Xe, ye, pe = Xte[chosen], yte[chosen], pred_te[chosen]

    # ---- GradientExplainer on the CNN -------------------------------------------------- #
    # Class-balanced background so rare classes are represented in the reference distribution.
    bg = torch.from_numpy(balanced_background(Xtr, ytr, 34, rng)).unsqueeze(1)
    explainer = shap.GradientExplainer(model, bg)
    sv = to_array(explainer.shap_values(torch.from_numpy(Xe).unsqueeze(1)))
    sv = sv.reshape(len(Xe), Xe.shape[1], 3)  # (n, time, class)
    t_ms = (np.arange(Xe.shape[1]) - HALF) / FS * 1000

    # ---- Fig 1: class profiles (correctly classified beats, SHAP for the true class) ---- #
    fig, axes = plt.subplots(3, 1, figsize=(8, 8.5), sharex=True, constrained_layout=True)
    for c, ax in enumerate(axes):
        ok = (ye == c) & (pe == c)
        if ok.sum() == 0:
            ax.set_title(f"{SHORT[c]}: no correctly classified beats in sample")
            continue
        wave = Xe[ok].mean(0)
        imp = np.abs(sv[ok][:, :, c]).mean(0)
        for (lo, hi), shade in zip(REGIONS.values(), ["#f3f4f6", "#e5e7eb", "#f3f4f6"]):
            ax.axvspan(t_ms[lo], t_ms[hi - 1], color=shade, zorder=0)
        ax.plot(t_ms, wave, color=INK, lw=1.6, label="mean beat (z-scored)")
        ax.set_ylabel("amplitude (z)")
        ax2 = ax.twinx()
        ax2.fill_between(t_ms, 0, imp, color=CLASS_COLORS[c], alpha=0.55, lw=0, label="mean |SHAP|")
        ax2.set_ylim(0, imp.max() * 1.25)
        ax2.set_ylabel("mean |SHAP|")
        ax.set_title(f"Class {SHORT[c]}  (n={ok.sum()} correctly classified)", loc="left", fontsize=11)
        if c == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=9)
    axes[-1].set_xlabel("time relative to R peak (ms)   |   grey bands: P/PR, QRS, ST/T (approximate)")
    fig.savefig(out / "shap_class_profiles.png", dpi=150)
    plt.close(fig)

    # ---- Fig 2: per-sample overlays: 3 correct examples + misclassified ------------------ #
    examples = []
    for c in range(3):
        ok = np.where((ye == c) & (pe == c))[0]
        if len(ok):
            examples.append((ok[0], "correct"))
    for i in np.where(pe != ye)[0][:3]:
        examples.append((i, "wrong"))
    ncol = 3
    nrow = int(np.ceil(len(examples) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.2 * nrow), constrained_layout=True,
                             squeeze=False)
    vmax = max(np.abs(sv[i, :, pe[i]]).max() for i, _ in examples)
    for ax in axes.ravel():
        ax.axis("off")
    sc = None
    for ax, (i, kind) in zip(axes.ravel(), examples):
        ax.axis("on")
        ax.plot(t_ms, Xe[i], color="#9ca3af", lw=1)
        sc = ax.scatter(t_ms, Xe[i], c=sv[i, :, pe[i]], cmap="coolwarm", vmin=-vmax, vmax=vmax,
                        s=14, zorder=3)
        ax.set_title(f"true {SHORT[ye[i]]} → pred {SHORT[pe[i]]} (p={proba(model, Xe[i:i+1])[0, pe[i]]:.2f})"
                     f"  [{kind}]", fontsize=10, loc="left", color=INK if kind == "correct" else "#b91c1c")
        ax.set_xlabel("ms from R peak")
        ax.set_ylabel("amplitude (z)")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.7, label="SHAP value for the predicted class\n"
                 "(red pushes toward it, blue pushes away)")
    fig.savefig(out / "shap_examples.png", dpi=150)
    plt.close(fig)

    # ---- Numeric check 1: where does the attribution mass sit? -------------------------- #
    region_rows, edge = {}, {}
    for c in range(3):
        ok = (ye == c) & (pe == c)
        if ok.sum() == 0:
            continue
        imp = np.abs(sv[ok][:, :, c]).mean(0)
        imp = imp / imp.sum()
        region_rows[SHORT[c]] = {name: float(imp[lo:hi].sum()) for name, (lo, hi) in REGIONS.items()}
        edge[SHORT[c]] = float(imp[:5].sum() + imp[-5:].sum())
    print("\nShare of mean |SHAP| per beat region (rows: true class, correctly classified beats)")
    for c, row in region_rows.items():
        print(f"  {c}: " + "  ".join(f"{k}: {v:.2f}" for k, v in row.items()) +
              f"   | first/last 5 samples: {edge[c]:.3f} (uniform would be {10 / Xe.shape[1]:.3f})")

    # ---- Numeric check 2: deletion test ------------------------------------------------- #
    ok = np.where(pe == ye)[0]
    k = args.k_mask
    drops_shap, drops_rand = [], []
    base = proba(model, Xe[ok])[np.arange(len(ok)), pe[ok]]
    Xm = Xe[ok].copy()
    for row, i in enumerate(ok):
        Xm[row, np.argsort(-np.abs(sv[i, :, pe[i]]))[:k]] = 0.0  # 0 == per-beat mean (z-scored)
    drops_shap = base - proba(model, Xm)[np.arange(len(ok)), pe[ok]]
    rand_drop = []
    for _ in range(10):
        Xr = Xe[ok].copy()
        for row in range(len(ok)):
            Xr[row, rng.choice(Xe.shape[1], k, replace=False)] = 0.0
        rand_drop.append(base - proba(model, Xr)[np.arange(len(ok)), pe[ok]])
    drops_rand = np.mean(rand_drop, axis=0)
    print(f"\nDeletion test (mask {k} of {Xe.shape[1]} samples, mean drop in predicted-class probability)")
    print(f"  top-SHAP samples: {drops_shap.mean():.3f}    random samples: {drops_rand.mean():.3f}")

    # ---- TreeSHAP on the baseline ------------------------------------------------------ #
    rf = joblib.load(out / "rf.joblib")
    Fte = extract_features(Xte)
    sub = rng.choice(len(Fte), min(500, len(Fte)), replace=False)
    tsv = to_array(shap.TreeExplainer(rf).shap_values(Fte[sub]))  # (n, feat, class)
    imp = np.abs(tsv).mean(0)  # (feat, class)
    order = np.argsort(-imp.sum(1))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    h = 0.26
    for c in range(3):
        ax.barh(np.arange(len(order)) + (c - 1) * h, imp[order, c], height=h,
                color=CLASS_COLORS[c], label=f"class {SHORT[c]}")
    ax.set_yticks(np.arange(len(order)), [FEATURE_NAMES[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP| (random forest, TreeSHAP)")
    ax.legend(frameon=False)
    fig.savefig(out / "shap_rf_features.png", dpi=150)
    plt.close(fig)

    json.dump({"region_share": region_rows, "edge_share_first_last_5": edge,
               "deletion_test": {"k": k, "top_shap_drop": float(drops_shap.mean()),
                                 "random_drop": float(drops_rand.mean())},
               "rf_top_features": [FEATURE_NAMES[i] for i in order[:5]]},
              open(out / "explain_metrics.json", "w"), indent=2)
    print("\nrandom-forest top features:", [FEATURE_NAMES[i] for i in order[:5]])
    print("figures written to", out)


if __name__ == "__main__":
    main()
