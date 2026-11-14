"""Train a feature-based baseline (random forest) and a 1D CNN with a subject-wise split.

    python train.py --data synthetic          # no download, pipeline smoke test
    python train.py --data mitbih             # real data (downloads from PhysioNet)
"""
import argparse
import json
import random
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import CLASS_NAMES, SHORT, extract_features, load_data, make_splits  # noqa: E402
from model import BeatCNN  # noqa: E402

BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#6b7280"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(y_true, y_pred):
    labels = [0, 1, 2]
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "per_class_recall": {CLASS_NAMES[i]: float(r) for i, r in enumerate(rec)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


@torch.no_grad()
def predict_proba(model, X, bs=1024):
    model.eval()
    out = [model(torch.from_numpy(X[i:i + bs]).unsqueeze(1)).softmax(1).numpy()
           for i in range(0, len(X), bs)]
    return np.concatenate(out)


def plot_confusions(results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(4.4 * len(results), 4), constrained_layout=True)
    for ax, (name, m) in zip(np.atleast_1d(axes), results.items()):
        cm = np.array(m["confusion_matrix"], dtype=float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{norm[i, j]:.2f}\n(n={int(cm[i, j])})", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "#111827", fontsize=9)
        ax.set_xticks(range(3), SHORT)
        ax.set_yticks(range(3), SHORT)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name}  (macro-F1 {m['macro_f1']:.3f})")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "mitbih"])
    ap.add_argument("--data-dir", default="data/mitdb")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    kw = {"seed": args.seed} if args.data == "synthetic" else {"data_dir": args.data_dir}
    X, y, pid = load_data(args.data, **kw)
    tr, va, te = make_splits(pid, args.data, args.seed)
    print(f"beats: train {len(tr)}  val {len(va)}  test {len(te)}  | patients "
          f"{len(set(pid[tr]))}/{len(set(pid[va]))}/{len(set(pid[te]))} (no overlap)")
    print("class counts train:", np.bincount(y[tr], minlength=3), " test:", np.bincount(y[te], minlength=3))

    # ---- baseline: handcrafted features + random forest -------------------------------- #
    F = extract_features(X)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1,
                                random_state=args.seed).fit(F[tr], y[tr])
    results = {"Random forest (features)": evaluate(y[te], rf.predict(F[te]))}
    joblib.dump(rf, out / "rf.joblib")

    # ---- 1D CNN ------------------------------------------------------------------------- #
    model = BeatCNN()
    counts = np.maximum(np.bincount(y[tr], minlength=3), 1)
    weights = torch.tensor(len(tr) / (3 * counts), dtype=torch.float32)  # inverse frequency
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(TensorDataset(torch.from_numpy(X[tr]).unsqueeze(1), torch.from_numpy(y[tr])),
                        batch_size=args.batch_size, shuffle=True)

    best_f1, history = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        sched.step()
        val_f1 = f1_score(y[va], predict_proba(model, X[va]).argmax(1), labels=[0, 1, 2],
                          average="macro", zero_division=0)
        history.append((total / len(tr), val_f1))
        if val_f1 > best_f1:  # keep the best epoch by VALIDATION macro-F1 (never by test)
            best_f1 = val_f1
            torch.save(model.state_dict(), out / "cnn.pt")
        print(f"epoch {ep:02d}  train loss {total / len(tr):.4f}  val macro-F1 {val_f1:.4f}")

    model.load_state_dict(torch.load(out / "cnn.pt"))
    results["1D CNN"] = evaluate(y[te], predict_proba(model, X[te]).argmax(1))

    # ---- outputs ------------------------------------------------------------------------ #
    json.dump({"dataset": args.data, "seed": args.seed, "best_val_macro_f1": best_f1,
               "test": results}, open(out / "metrics.json", "w"), indent=2)
    np.savez_compressed(out / "arrays.npz", X_train=X[tr], y_train=y[tr], X_test=X[te],
                        y_test=y[te], pid_test=pid[te])
    plot_confusions(results, out / "confusion_matrices.png")

    h = np.array(history)
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    ax.plot(h[:, 0], color=BLUE, lw=2, label="train loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax2 = ax.twinx()  # same x, different quantity: only used for this training-curve view
    ax2.plot(h[:, 1], color=ORANGE, lw=2, label="val macro-F1")
    ax2.set_ylabel("validation macro-F1")
    ax2.set_ylim(0, 1.02)
    fig.legend(loc="lower center", ncol=2, frameon=False)
    fig.savefig(out / "training_curve.png", dpi=150)
    plt.close(fig)

    print("\nTEST RESULTS (patients never seen in training)")
    for name, m in results.items():
        rec = ", ".join(f"{k.split()[0]} {v:.3f}" for k, v in m["per_class_recall"].items())
        print(f"  {name:28s} macro-F1 {m['macro_f1']:.3f}  acc {m['accuracy']:.3f}  recall: {rec}")


if __name__ == "__main__":
    main()
