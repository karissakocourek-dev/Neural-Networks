"""
Wearable Activity Detective - train and evaluate script.

Question: can a small neural network tell WALKING, WALKING_UPSTAIRS and
WALKING_DOWNSTAIRS apart from smartphone sensor data?

Data: UCI "Human Activity Recognition Using Smartphones" (Anguita et al., 2013),
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

Usage:
    python train_and_evaluate.py "<path to UCI HAR Dataset folder>" <output folder>

Outputs (in the output folder):
    results.json   metrics, settings, chosen mistake
    gallery.json   held-out test windows with raw accelerometer trace + predictions
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42
rng = np.random.default_rng(SEED)

data_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

# The three walking activities in the dataset's label codes.
WALKING_CODES = {1: "Walking", 2: "Walking upstairs", 3: "Walking downstairs"}
class_names = list(WALKING_CODES.values())


def load_split(split):
    x = np.loadtxt(data_dir / split / f"X_{split}.txt")
    y = np.loadtxt(data_dir / split / f"y_{split}.txt", dtype=int)
    subject = np.loadtxt(data_dir / split / f"subject_{split}.txt", dtype=int)
    keep = np.isin(y, list(WALKING_CODES))
    # Raw total acceleration (gravity + body motion), 128 readings at 50 Hz, x/y/z.
    signals = np.stack(
        [
            np.loadtxt(data_dir / split / "Inertial Signals" / f"total_acc_{axis}_{split}.txt")
            for axis in "xyz"
        ],
        axis=1,
    )
    labels = np.array([list(WALKING_CODES).index(v) for v in y[keep]])
    return x[keep], labels, subject[keep], signals[keep]


x_all_train, y_all_train, subj_train, _ = load_split("train")
x_test, y_test, subj_test, sig_test = load_split("test")

# ---- Split the training subjects into train / validation (by person, not by window) ----
train_subjects = np.unique(subj_train)
val_subjects = rng.choice(train_subjects, size=5, replace=False)
is_val = np.isin(subj_train, val_subjects)
x_tr, y_tr = x_all_train[~is_val], y_all_train[~is_val]
x_val, y_val = x_all_train[is_val], y_all_train[is_val]

scaler = StandardScaler().fit(x_tr)

# ---- Tune the network on the validation subjects only ----
candidates = []
for hidden in [(16,), (32,), (64,), (64, 32)]:
    for alpha in [1e-4, 1e-2, 1.0]:
        model = MLPClassifier(
            hidden_layer_sizes=hidden, alpha=alpha, max_iter=300, random_state=SEED
        ).fit(scaler.transform(x_tr), y_tr)
        val_acc = accuracy_score(y_val, model.predict(scaler.transform(x_val)))
        candidates.append({"hidden": list(hidden), "alpha": alpha, "val_accuracy": round(val_acc, 4)})
best = max(candidates, key=lambda c: c["val_accuracy"])

# ---- Refit the chosen settings on all training subjects, then touch the test set once ----
scaler = StandardScaler().fit(x_all_train)
net = MLPClassifier(
    hidden_layer_sizes=tuple(best["hidden"]), alpha=best["alpha"], max_iter=300, random_state=SEED
).fit(scaler.transform(x_all_train), y_all_train)

baseline = DummyClassifier(strategy="most_frequent").fit(x_all_train, y_all_train)
logreg = LogisticRegression(max_iter=2000).fit(scaler.transform(x_all_train), y_all_train)

xt = scaler.transform(x_test)
pred_net = net.predict(xt)
proba_net = net.predict_proba(xt)
pred_base = baseline.predict(x_test)
pred_lr = logreg.predict(xt)

acc_net = accuracy_score(y_test, pred_net)
acc_base = accuracy_score(y_test, pred_base)
acc_lr = accuracy_score(y_test, pred_lr)
cm = confusion_matrix(y_test, pred_net, labels=[0, 1, 2])

# Per-subject test accuracy shows how much results vary between people.
per_subject = {
    int(s): round(float(accuracy_score(y_test[subj_test == s], pred_net[subj_test == s])), 3)
    for s in np.unique(subj_test)
}

# ---- Choose one mistake to highlight: the wrong prediction the net was most sure about ----
wrong = np.where(pred_net != y_test)[0]
conf = proba_net[np.arange(len(y_test)), pred_net]
mistake_idx = int(wrong[np.argmax(conf[wrong])]) if len(wrong) else None

# ---- Gallery: all mistakes plus a random sample of correct windows, shuffled ----
n_correct = 120
correct = np.where(pred_net == y_test)[0]
chosen = np.concatenate([wrong, rng.choice(correct, size=min(n_correct, len(correct)), replace=False)])
rng.shuffle(chosen)
gallery = []
for i in chosen:
    gallery.append(
        {
            "id": int(i),
            "subject": int(subj_test[i]),
            "truth": int(y_test[i]),
            "pred": int(pred_net[i]),
            "proba": [round(float(p), 3) for p in proba_net[i]],
            "x": np.round(sig_test[i, 0], 2).tolist(),
            "y": np.round(sig_test[i, 1], 2).tolist(),
            "z": np.round(sig_test[i, 2], 2).tolist(),
        }
    )

results = {
    "data_source": "UCI Human Activity Recognition Using Smartphones (Anguita et al., 2013)",
    "classes": class_names,
    "n_train_windows": int(len(y_all_train)),
    "n_train_subjects": int(len(train_subjects)),
    "n_test_windows": int(len(y_test)),
    "n_test_subjects": int(len(np.unique(subj_test))),
    "test_class_counts": {class_names[k]: int((y_test == k).sum()) for k in range(3)},
    "n_features": int(x_all_train.shape[1]),
    "validation_subjects": [int(s) for s in sorted(val_subjects)],
    "tuning_candidates": candidates,
    "chosen": best,
    "test_accuracy_network": round(float(acc_net), 4),
    "test_accuracy_baseline_most_frequent": round(float(acc_base), 4),
    "baseline_predicts": class_names[int(baseline.predict(x_test[:1])[0])],
    "test_accuracy_logistic_regression": round(float(acc_lr), 4),
    "n_test_mistakes": int(len(wrong)),
    "confusion_matrix_rows_true_cols_pred": cm.tolist(),
    "per_subject_test_accuracy": per_subject,
    "gallery_size": len(gallery),
    "gallery_mistakes": int(len(wrong)),
    "gallery_correct_sample": int(len(chosen) - len(wrong)),
    "highlighted_mistake": None
    if mistake_idx is None
    else {
        "id": mistake_idx,
        "subject": int(subj_test[mistake_idx]),
        "truth": class_names[int(y_test[mistake_idx])],
        "predicted": class_names[int(pred_net[mistake_idx])],
        "confidence": round(float(conf[mistake_idx]), 3),
    },
}

(out_dir / "results.json").write_text(json.dumps(results, indent=2))
(out_dir / "gallery.json").write_text(json.dumps({"classes": class_names, "windows": gallery}, separators=(",", ":")))
print(json.dumps({k: v for k, v in results.items() if k != "per_subject_test_accuracy"}, indent=2))
print("per subject:", per_subject)
