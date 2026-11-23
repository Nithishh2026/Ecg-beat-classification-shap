# Explainable ECG beat classification (PyTorch + SHAP)

Classify single heartbeats into **N** (normal), **S** (supraventricular ectopic) and
**V** (ventricular ectopic), compare a 1D CNN with a feature-based random forest, and
use SHAP to check *where in the beat* each model looks.

## Quick start

```bash
pip install -r requirements.txt

# 1) Pipeline smoke test (no download, synthetic beats)
python train.py --data synthetic
python explain.py --out results

# 2) Real run on MIT-BIH (downloads ~100 MB from PhysioNet on first use)
python train.py --data mitbih --out results_mitbih
python explain.py --out results_mitbih
```

## What is in the repo

| File | Purpose |
|---|---|
| `data.py` | MIT-BIH loader (wfdb), synthetic beat generator, band-pass + per-beat normalisation, handcrafted features, **subject-wise splits** |
| `model.py` | Small 1D CNN (3 conv blocks, BatchNorm, `ReLU(inplace=False)`, flatten head) |
| `train.py` | Random forest baseline + CNN with weighted loss; best epoch chosen on validation macro-F1; writes metrics and confusion matrices |
| `explain.py` | GradientExplainer (CNN), TreeSHAP (baseline), region-attribution table, deletion test |

## Design decisions

- **Split by patient, never by beat.** Beats from one patient are highly similar; a random split leaks. MIT-BIH uses the fixed DS1/DS2 inter-patient protocol (train/val from DS1, test = DS2); `data.py` asserts no patient appears in two sets.
- **Class imbalance:** inverse-frequency loss weights for the CNN, `class_weight="balanced"` for the forest. Report **macro-F1 and per-class recall**, not accuracy.
- **Model selection on validation only.** The test set is touched once, at the end.
- **SHAP background:** class-balanced sample of training beats, so rare classes are represented in the reference distribution.

## How SHAP is checked, not just plotted

1. `shap_class_profiles.png`: mean beat with mean |SHAP| overlaid, per class.
2. `shap_examples.png`: per-sample SHAP on the waveform for correct and misclassified beats.
3. `shap_rf_features.png`: TreeSHAP on the baseline, to compare what the two models rely on.
4. **Region attribution** (`explain_metrics.json`): share of |SHAP| on P/PR, QRS and ST/T per class, plus the share on the first/last 5 samples (high values suggest an edge artifact).
5. **Deletion test:** masking the top-k SHAP samples should reduce the predicted-class probability far more than masking k random samples.

## Results

### Synthetic smoke test (pipeline check only, NOT a scientific result)

The synthetic generator has clean, known morphology (S = absent/weak P wave, V = wide QRS and inverted T), so it is easy. It exists to prove the code runs and to give SHAP a ground truth to be compared against.

| Model | Macro-F1 | Recall N / S / V |
|---|---|---|
| Random forest (features) | 0.894 | 0.919 / 0.943 / 1.000 |
| 1D CNN | 0.998 | 0.999 / 0.994 / 1.000 |

SHAP on synthetic data: N and S attributions concentrate on the P/PR region (62% and 58%), V on the QRS (49%); deletion test drop 0.140 (top-SHAP) vs 0.001 (random).

### MIT-BIH (fill in after your run)

| Model | Macro-F1 | Recall N / S / V |
|---|---|---|
| Random forest (features) | _TBD_ | _TBD_ |
| 1D CNN | _TBD_ | _TBD_ |

Add 2-3 sentences: which class is hardest (usually S), what SHAP shows for it, and whether that matches physiology (S beats: abnormal P wave / timing; V beats: wide QRS, discordant T).

## Limitations

- MIT-BIH has only 47 patients and one lead (MLII); results may not transfer to other devices or populations.
- Beat-level classification ignores rhythm context (RR intervals), which matters clinically for S beats.
- SHAP attributions describe the model, not physiology or causality; masked inputs in the deletion test are out of distribution.
- Region boundaries are approximate fixed sample ranges, not per-beat delineation.

## Ideas to extend

- Add RR-interval features (previous/next beat spacing) to both models.
- Compare 5 random seeds and report mean ± std.
- Try WESAD (stress, EDA/ECG) with the same pipeline: swap the loader, keep the rest.
- Add a notebook that walks through one beat from raw signal to explanation.
