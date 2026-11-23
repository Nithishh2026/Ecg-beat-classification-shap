# ECG beat classification with a CNN, and what SHAP says about it

This project classifies single heartbeats into three classes: N (normal), S (supraventricular ectopic) and V (ventricular ectopic). I train a small 1D CNN and a random forest on handcrafted features, then use SHAP to see which part of the beat each model actually looks at. The question I care about is whether the models focus on the parts of the waveform a cardiologist would.

## Running it

```bash
pip install -r requirements.txt

# quick check that everything runs (synthetic beats, nothing to download)
python train.py --data synthetic
python explain.py --out results

# the real thing on MIT-BIH (about 100 MB from PhysioNet the first time)
python train.py --data mitbih --out results_mitbih
python explain.py --out results_mitbih
```

## Files

- `data.py`: loads MIT-BIH through wfdb, generates synthetic beats, does the band-pass filter and per-beat normalisation, computes the handcrafted features, and makes the patient-wise splits.
- `model.py`: the 1D CNN. Three conv blocks with BatchNorm, `ReLU(inplace=False)` (SHAP's gradient hooks complain otherwise), then a flatten head.
- `train.py`: trains the random forest baseline and the CNN. The CNN uses a weighted loss and I keep the epoch with the best validation macro-F1. Writes metrics and confusion matrices.
- `explain.py`: GradientExplainer for the CNN, TreeSHAP for the forest, a table of attribution per beat region, and a deletion test.

## Choices worth explaining

**Splitting by patient.** Beats from the same person look very alike, so a random beat-level split leaks information and gives inflated scores. For MIT-BIH I use the usual DS1/DS2 inter-patient split: train and validation come from DS1, and the test set is DS2. `data.py` has an assert that no patient shows up in two sets.

**Class imbalance.** N dominates by a lot. The CNN gets inverse-frequency loss weights and the forest uses `class_weight="balanced"`. Because accuracy is meaningless here, I report macro-F1 and per-class recall.

**Model selection.** Everything is picked on the validation set. The test set is only used once, at the very end.

**SHAP background.** I draw a class-balanced sample of training beats as the background, otherwise the rare classes would barely be represented.

## Checking the SHAP results instead of just trusting the plots

Pretty SHAP plots are easy to over-read, so I added a few sanity checks:

1. `shap_class_profiles.png` shows the mean beat per class with the mean |SHAP| drawn over it.
2. `shap_examples.png` shows per-sample attributions on the waveform, for both correct and misclassified beats.
3. `shap_rf_features.png` is TreeSHAP on the random forest, so I can compare what the two models rely on.
4. The region attribution in `explain_metrics.json` gives the share of |SHAP| that falls on P/PR, QRS and ST/T for each class. It also reports the share on the first and last 5 samples, since a high value there usually means an edge artifact.
5. The deletion test masks the top-k SHAP samples and checks that the predicted-class probability drops much more than when k random samples are masked.

## Results

### Synthetic data (just a pipeline check)

The synthetic beats have clean, known shapes (S has a missing or weak P wave, V has a wide QRS and an inverted T), so the task is easy and these numbers say nothing scientific. The point is to confirm the code works and to have a case where I know what SHAP should find.

| Model | Macro-F1 | Recall N / S / V |
|---|---|---|
| Random forest (features) | 0.894 | 0.919 / 0.943 / 1.000 |
| 1D CNN | 0.998 | 0.999 / 0.994 / 1.000 |

SHAP behaves sensibly here. For N and S most of the attribution sits on the P/PR region (62% and 58%) and for V it sits on the QRS (49%). In the deletion test, masking the top SHAP samples lowers the probability by 0.140, against 0.001 for random masking.

### MIT-BIH (to be filled in after my run)

| Model | Macro-F1 | Recall N / S / V |
|---|---|---|
| Random forest (features) | TBD | TBD |
| 1D CNN | TBD | TBD |

Still to write: which class is hardest (I expect S), what SHAP shows for it, and whether that fits the physiology. S beats should show an abnormal P wave or odd timing, and V beats a wide QRS with a discordant T wave.

## Limitations

- MIT-BIH only has 47 patients and a single lead (MLII), so the results may not carry over to other devices or populations.
- The classifier looks at one beat at a time and ignores rhythm context such as RR intervals, which matters clinically, especially for S beats.
- SHAP explains the model, not the physiology, and it says nothing about causality. The masked inputs in the deletion test are also out of distribution.
- The P/PR, QRS and ST/T regions are fixed sample ranges, not a per-beat delineation, so they are only approximate.

## Possible next steps

- Add RR-interval features (spacing to the previous and next beat) to both models.
- Run 5 random seeds and report mean and standard deviation.
- Try WESAD (stress, EDA/ECG) by swapping the loader and keeping the rest of the pipeline.
- Write a notebook that follows one beat from the raw signal to its explanation.
