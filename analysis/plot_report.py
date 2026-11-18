"""Regenerate report charts from archived and re-evaluated Team 9 results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analysis" / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

COLORS = {
    "XGBoost": "#0891b2", "MLP": "#7c3aed", "Random Forest": "#16a34a",
    "SVM": "#f59e0b", "KNN": "#e11d48",
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def load() -> dict:
    tuning = ROOT / "outputs" / "xgboost" / "hyperparameter_tuning"
    cv = pd.read_csv(tuning / "cv_results.csv")
    folds = cv[pd.to_numeric(cv.fold, errors="coerce").notna()].copy()
    folds.macro_f1 = folds.macro_f1.astype(float)
    xgb = json.loads((RESULTS / "xgboost_verified.json").read_text())
    mlp = json.loads((RESULTS / "mlp_verified.json").read_text())
    mlp_run = json.loads((ROOT / "analysis" / "mlp_shared" / "metrics.json").read_text())
    data = {
        "XGBoost": {**xgb, "cv_f1": float(folds.macro_f1.mean()),
                    "cv_sd": float(folds.macro_f1.std(ddof=1))},
        "MLP": {**mlp,
                "cv_f1": mlp_run["cross_validation_mean_and_sd"]["macro_f1"]["mean"],
                "cv_sd": mlp_run["cross_validation_mean_and_sd"]["macro_f1"]["std"]},
    }
    for name, file in [("Random Forest", "random_forest.json"), ("SVM", "svm.json"),
                       ("KNN", "knn.json")]:
        result = json.loads((RESULTS / file).read_text())
        data[name] = {**result["student_test"],
                      "cv_f1": result["cv_macro_f1_mean"],
                      "cv_sd": result["cv_macro_f1_sd"]}
    return data


def save(fig, name: str):
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def model_charts(data: dict):
    names = list(data)
    positions = np.arange(len(names))
    colors = [COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(9.4, 4.5), layout="constrained")
    vals = [data[n]["cv_f1"] for n in names]
    errors = [data[n]["cv_sd"] for n in names]
    ax.bar(positions, vals, color=colors, alpha=.9)
    ax.errorbar(positions, vals, yerr=errors, fmt="none", ecolor="#334155", capsize=5)
    ax.set_xticks(positions, names)
    ax.set_ylim(0.60, 0.92)
    ax.set_ylabel("Candidate Macro F1")
    ax.set_title("Five frozen trip folds - mean with sample SD", loc="left", weight="bold")
    ax.grid(axis="y", color="#e2e8f0")
    ax.set_axisbelow(True)
    for i, value in enumerate(vals):
        ax.text(i, value + .012, f"{value:.3f}", ha="center", fontsize=9)
    save(fig, "cv_macro_f1.png")

    fig, ax = plt.subplots(figsize=(9.4, 4.5), layout="constrained")
    width = .36
    top1 = [data[n]["conditional_top1"] for n in names]
    f1 = [data[n]["candidate_macro_f1"] for n in names]
    ax.bar(positions - width / 2, top1, width, color="#0284c7", label="Conditional Top-1")
    ax.bar(positions + width / 2, f1, width, color="#0f766e", label="Candidate Macro F1")
    ax.set_xticks(positions, names)
    ax.set_ylim(0.60, 0.94)
    ax.set_ylabel("Student-test score")
    ax.set_title("Held-out student-test: two different scoring units", loc="left", weight="bold")
    ax.grid(axis="y", color="#e2e8f0")
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    save(fig, "student_test_top1_f1.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4), layout="constrained")
    losses = [(n, data[n]["log_loss"]) for n in names if data[n]["log_loss"] is not None]
    axes[0].bar([x[0] for x in losses], [x[1] for x in losses],
                color=[COLORS[x[0]] for x in losses])
    axes[0].set_ylim(0, max(x[1] for x in losses) * 1.2)
    axes[0].set_title("Student-test log loss", loc="left", weight="bold")
    axes[0].set_ylabel("Lower is better")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(names, [data[n]["classifier_route_continuity"] for n in names],
                color=colors)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_title("Classifier-only route continuity", loc="left", weight="bold")
    axes[1].set_ylabel("Same arc or direct endpoint")
    axes[1].tick_params(axis="x", rotation=20)
    for ax in axes:
        ax.grid(axis="y", color="#e2e8f0")
        ax.set_axisbelow(True)
    save(fig, "loss_continuity.png")


def hmm_chart():
    d = pd.read_csv(RESULTS / "hmm_ablation.csv", dtype={"trip_id": str})
    fig, ax = plt.subplots(figsize=(9.4, 4.5), layout="constrained")
    x = np.arange(len(d))
    ax.plot(x, d.classifier_arc_f1, marker="o", color="#64748b", lw=1.8,
            label="Independent XGBoost choices")
    ax.plot(x, d.hmm_arc_f1, marker="o", color="#0891b2", lw=2.2,
            label="XGBoost + HMM route")
    ax.set_xticks(x, d.trip_id.str[-2:])
    ax.set_xlabel("Evaluator trip suffix")
    ax.set_ylabel("Unique route arc F1")
    ax.set_ylim(.45, 1.02)
    ax.set_title("Evaluator trips - reference arc set overlap", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e2e8f0")
    ax.set_axisbelow(True)
    save(fig, "hmm_ablation.png")


def mlp_curve():
    base = ROOT / "analysis" / "mlp_shared" / "histories"
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4), layout="constrained")
    for i in range(1, 6):
        d = pd.read_csv(base / f"fold_{i}.csv")
        axes[0].plot(d.epoch, d.training_loss, label=f"Fold {i}", lw=1.4)
        axes[1].plot(d.epoch, d.validation_log_loss, label=f"Fold {i}", lw=1.4)
    axes[0].set_title("MLP training objective", loc="left", weight="bold")
    axes[1].set_title("Held-out validation log loss", loc="left", weight="bold")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(color="#e2e8f0")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Loss")
    axes[1].legend(ncol=2, fontsize=8, frameon=False)
    save(fig, "mlp_learning_curves.png")


if __name__ == "__main__":
    model_charts(load())
    hmm_chart()
    mlp_curve()
    print(FIGURES)
