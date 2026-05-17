"""
All visualisation routines: EDA charts, preprocessing diagnostics,
and model evaluation plots.

Visual theme: clean clinical style with a consistent medical palette.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from config import TARGET_COL, REPORTS_DIR

# ── Design Tokens ─────────────────────────────────────────────────────────────
_C0   = "#2C82BE"   # No-CHD blue
_C1   = "#C0392B"   # CHD red
_NAVY = "#1A2340"   # titles / axes
_GRID = "#E8ECF0"   # gridlines
_BG   = "#FFFFFF"
_FACE = "#F7F9FC"   # axis background

PALETTE = {"0": _C0, "1": _C1}
_MODEL_PALETTE = [
    "#2C82BE", "#C0392B", "#27AE60", "#8E44AD",
    "#E67E22", "#16A085", "#2C3E50", "#D35400", "#7F8C8D",
]


def _apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor":  _BG,
        "axes.facecolor":    _FACE,
        "axes.edgecolor":    "#D0D8E0",
        "axes.labelcolor":   _NAVY,
        "axes.titlecolor":   _NAVY,
        "axes.titleweight":  "bold",
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "axes.grid":         True,
        "grid.color":        _GRID,
        "grid.linewidth":    0.8,
        "xtick.color":       _NAVY,
        "ytick.color":       _NAVY,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.frameon":    True,
        "legend.framealpha": 0.9,
        "legend.fontsize":   9,
        "font.family":       "DejaVu Sans",
        "text.color":        _NAVY,
    })


_apply_theme()


def _clean_spines(ax, keep=("bottom", "left")) -> None:
    for spine in ax.spines:
        ax.spines[spine].set_visible(spine in keep)


def _save(fig, name: str) -> None:
    path = os.path.join(REPORTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=180, facecolor=_BG)
    plt.close(fig)
    print(f"Saved -> {path}")


def _legend_patches():
    return [
        mpatches.Patch(color=_C0, label="No CHD"),
        mpatches.Patch(color=_C1, label="CHD"),
    ]


# ── EDA ───────────────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET_COL].value_counts().sort_index()
    labels = ["No CHD", "CHD"]
    colors = [_C0, _C1]
    total  = counts.sum()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                              gridspec_kw={"wspace": 0.35})
    fig.patch.set_facecolor(_BG)
    fig.suptitle("Target Variable: 10-Year Coronary Heart Disease Risk",
                 fontsize=15, fontweight="bold", color=_NAVY, y=1.02)

    # — Donut chart —
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        counts, labels=None, autopct="%1.1f%%",
        colors=colors, startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
        at.set_color("white")
    ax.set_title("Class Balance", pad=12)
    ax.legend(handles=_legend_patches(), loc="lower center",
              bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.text(0, 0, f"n = {total:,}", ha="center", va="center",
            fontsize=11, fontweight="bold", color=_NAVY)

    # — Annotated bar chart —
    ax2 = axes[1]
    bars = ax2.bar(labels, counts.values, color=colors,
                   edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + total * 0.01,
                 f"{val:,}\n({val/total*100:.1f}%)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold",
                 color=_NAVY)
    ax2.set_title("Class Counts", pad=12)
    ax2.set_ylabel("Number of Patients")
    ax2.set_ylim(0, counts.max() * 1.18)
    _clean_spines(ax2)
    ax2.grid(axis="y", color=_GRID, linewidth=0.8)
    ax2.grid(axis="x", visible=False)

    _save(fig, "01_class_distribution.png")


def plot_missing_heatmap(df: pd.DataFrame) -> None:
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5),
                              gridspec_kw={"width_ratios": [3, 1], "wspace": 0.04})
    fig.suptitle("Missing Value Analysis", fontsize=15,
                 fontweight="bold", color=_NAVY, y=1.02)

    # Heatmap
    cmap = LinearSegmentedColormap.from_list("miss", ["#EAF4FC", "#C0392B"])
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False,
                cmap=cmap, ax=axes[0])
    axes[0].set_title("Row-level Missing Pattern", pad=10)
    axes[0].set_xlabel("Feature")

    # Bar summary
    if len(missing) > 0:
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(missing)))
        bars = axes[1].barh(missing.index, missing.values,
                            color=colors, edgecolor="white")
        for bar, val in zip(bars, missing.values):
            axes[1].text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                         f"{val:.1f}%", va="center", fontsize=9,
                         color=_NAVY, fontweight="bold")
        axes[1].set_xlabel("Missing %")
        axes[1].set_title("% Missing per Feature", pad=10)
        axes[1].set_xlim(0, missing.max() * 1.35)
        _clean_spines(axes[1])
        axes[1].invert_yaxis()
    else:
        axes[1].text(0.5, 0.5, "No missing\nvalues", ha="center",
                     va="center", fontsize=12, color=_NAVY,
                     transform=axes[1].transAxes)
        axes[1].axis("off")

    _save(fig, "02_missing_values.png")


def plot_continuous_distributions(df: pd.DataFrame, cont_cols: list) -> None:
    n = len(cont_cols)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.2))
    axes = np.array(axes).flatten()

    fig.suptitle("Continuous Feature Distributions by CHD Risk",
                 fontsize=15, fontweight="bold", color=_NAVY, y=1.01)

    for i, col in enumerate(cont_cols):
        ax = axes[i]
        for label, color in zip([0, 1], [_C0, _C1]):
            subset = df[df[TARGET_COL] == label][col].dropna()
            ax.hist(subset, bins=30, alpha=0.55, color=color,
                    edgecolor="none", density=True)
            subset.plot.kde(ax=ax, color=color, linewidth=2)
            ax.axvline(subset.mean(), color=color, linestyle="--",
                       linewidth=1.2, alpha=0.8)
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("")
        _clean_spines(ax)
        ax.grid(axis="y", color=_GRID)
        ax.grid(axis="x", visible=False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.legend(handles=_legend_patches(), loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.02), fontsize=10)
    fig.tight_layout()
    _save(fig, "03_continuous_distributions.png")


def plot_categorical_counts(df: pd.DataFrame, cat_cols: list) -> None:
    n = len(cat_cols)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = np.array(axes).flatten()

    fig.suptitle("Categorical Feature Counts by CHD Risk",
                 fontsize=15, fontweight="bold", color=_NAVY, y=1.01)

    for i, col in enumerate(cat_cols):
        ax = axes[i]
        ct = df.groupby([col, TARGET_COL]).size().unstack(fill_value=0)
        ct.plot(kind="bar", ax=ax, color=[_C0, _C1],
                edgecolor="white", linewidth=1, width=0.6)
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.legend().remove()
        _clean_spines(ax)
        ax.grid(axis="y", color=_GRID)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.legend(handles=_legend_patches(), loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.02), fontsize=10)
    fig.tight_layout()
    _save(fig, "04_categorical_counts.png")


def plot_boxplots(df: pd.DataFrame, cont_cols: list) -> None:
    n = len(cont_cols)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = np.array(axes).flatten()

    fig.suptitle("Feature Distributions vs CHD Risk (Boxplots)",
                 fontsize=15, fontweight="bold", color=_NAVY, y=1.01)

    for i, col in enumerate(cont_cols):
        ax = axes[i]
        groups = [df[df[TARGET_COL] == k][col].dropna().values
                  for k in [0, 1]]
        bp = ax.boxplot(groups, patch_artist=True,
                        medianprops={"color": "white", "linewidth": 2},
                        whiskerprops={"linewidth": 1.2},
                        capprops={"linewidth": 1.2},
                        flierprops={"marker": "o", "markersize": 3,
                                    "alpha": 0.4})
        for patch, color in zip(bp["boxes"], [_C0, _C1]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticklabels(["No CHD", "CHD"])
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        _clean_spines(ax)
        ax.grid(axis="y", color=_GRID)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    _save(fig, "05_boxplots.png")


def plot_correlation_heatmap(df: pd.DataFrame, title_suffix: str = "") -> None:
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 11))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap=cmap, center=0, vmin=-1, vmax=1,
        linewidths=0.4, linecolor="#DDE3EA",
        annot_kws={"size": 8},
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    label = f" — {title_suffix}" if title_suffix else ""
    ax.set_title(f"Feature Correlation Matrix{label}", pad=15)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    _save(fig, f"06_correlation_heatmap{'_' + title_suffix if title_suffix else ''}.png")


# ── Model Evaluation ──────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    from sklearn.metrics import confusion_matrix as sk_cm
    cm = sk_cm(y_true, y_pred)
    total = cm.sum()
    pct   = cm / total * 100

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap=LinearSegmentedColormap.from_list(
        "cm_cmap", ["#EAF4FC", _C0]), aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nNo CHD", "Predicted\nCHD"], fontsize=10)
    ax.set_yticklabels(["Actual\nNo CHD", "Actual\nCHD"], fontsize=10)

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else _NAVY
            ax.text(j, i, f"{cm[i, j]:,}\n({pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color=color)

    ax.set_title(f"Confusion Matrix — {model_name}", pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    _clean_spines(ax, keep=())
    _save(fig, f"cm_{model_name.replace(' ', '_')}.png")


def plot_roc_curves(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    colors = _MODEL_PALETTE
    aucs = {}
    for (name, data), color in zip(results.items(), colors):
        if data.get("y_prob") is None:
            continue
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        roc_auc = auc(fpr, tpr)
        aucs[name] = roc_auc
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name}  AUC = {roc_auc:.3f}")

    # Shade area under best model
    best_name = max(aucs, key=aucs.get) if aucs else None
    if best_name:
        best_data = results[best_name]
        fpr_b, tpr_b, _ = roc_curve(best_data["y_true"], best_data["y_prob"])
        best_idx = list(results.keys()).index(best_name)
        ax.fill_between(fpr_b, tpr_b, alpha=0.08,
                        color=colors[best_idx % len(colors)])

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("ROC Curves — All Models", pad=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    # Annotate best AUC
    if best_name:
        ax.annotate(f"Best: {best_name}\nAUC = {aucs[best_name]:.3f}",
                    xy=(0.6, 0.15), fontsize=9, color=_NAVY,
                    bbox={"boxstyle": "round,pad=0.4", "fc": "#EAF4FC",
                          "ec": _C0, "alpha": 0.9})

    _clean_spines(ax)
    _save(fig, "07_roc_curves_all_models.png")


def plot_metrics_comparison(metrics_df: pd.DataFrame) -> None:
    metric_cols = [c for c in ["Accuracy", "Precision", "Recall",
                                "F1", "ROC_AUC"] if c in metrics_df.columns]
    plot_df = metrics_df.copy()
    if "Model" not in plot_df.columns:
        plot_df = plot_df.reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5),
                              gridspec_kw={"width_ratios": [2, 1], "wspace": 0.35})
    fig.suptitle("Model Performance Comparison", fontsize=15,
                 fontweight="bold", color=_NAVY, y=1.02)

    # — Grouped bar chart —
    melted = plot_df.melt(id_vars="Model", value_vars=metric_cols,
                          var_name="Metric", value_name="Score")
    ax = axes[0]
    models  = plot_df["Model"].tolist()
    n_mod   = len(models)
    n_met   = len(metric_cols)
    x       = np.arange(n_met)
    width   = 0.8 / n_mod

    for idx, (model, color) in enumerate(zip(models, _MODEL_PALETTE)):
        offsets = (idx - n_mod / 2 + 0.5) * width
        scores  = [plot_df[plot_df["Model"] == model][m].values[0]
                   for m in metric_cols]
        bars = ax.bar(x + offsets, scores, width * 0.9,
                      label=model, color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.8)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{score:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color=_NAVY, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title("All Metrics by Model")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    _clean_spines(ax)
    ax.grid(axis="y", color=_GRID)
    ax.grid(axis="x", visible=False)

    # — Heatmap summary —
    ax2 = axes[1]
    heat_data = plot_df.set_index("Model")[metric_cols]
    cmap = LinearSegmentedColormap.from_list("perf", ["#EAF4FC", _C0])
    sns.heatmap(heat_data, annot=True, fmt=".3f",
                cmap=cmap, vmin=0.5, vmax=1.0,
                linewidths=0.4, linecolor="#DDE3EA",
                annot_kws={"size": 9, "weight": "bold"},
                cbar_kws={"shrink": 0.8},
                ax=ax2)
    ax2.set_title("Score Heatmap")
    ax2.tick_params(axis="x", rotation=30, labelsize=9)
    ax2.tick_params(axis="y", rotation=0, labelsize=9)

    _save(fig, "08_metrics_comparison.png")


def plot_feature_importance(importances: pd.Series, model_name: str) -> None:
    imp = importances.sort_values(ascending=True)
    n   = len(imp)

    cmap   = LinearSegmentedColormap.from_list("imp", ["#AED6F1", _C0])
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.38)))
    bars = ax.barh(imp.index, imp.values, color=colors,
                   edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, imp.values):
        ax.text(val + imp.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8.5, color=_NAVY)

    ax.set_title(f"Feature Importance — {model_name}", pad=12)
    ax.set_xlabel("Importance Score")
    ax.set_xlim(0, imp.max() * 1.18)
    _clean_spines(ax)
    ax.grid(axis="x", color=_GRID)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, f"09_feature_importance_{model_name.replace(' ', '_')}.png")


def plot_learning_curve(train_scores: np.ndarray,
                        val_scores: np.ndarray,
                        train_sizes: np.ndarray,
                        model_name: str) -> None:
    tr_mean = train_scores.mean(axis=1)
    tr_std  = train_scores.std(axis=1)
    va_mean = val_scores.mean(axis=1)
    va_std  = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(train_sizes, tr_mean, "o-", color=_C0, linewidth=2,
            label="Train Score", markersize=5)
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std,
                    alpha=0.15, color=_C0)

    ax.plot(train_sizes, va_mean, "s-", color=_C1, linewidth=2,
            label="Validation Score", markersize=5)
    ax.fill_between(train_sizes, va_mean - va_std, va_mean + va_std,
                    alpha=0.15, color=_C1)

    gap = tr_mean[-1] - va_mean[-1]
    ax.annotate(f"Gap = {gap:.3f}", xy=(train_sizes[-1], va_mean[-1]),
                xytext=(-60, 15), textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": _NAVY},
                fontsize=9, color=_NAVY,
                bbox={"boxstyle": "round,pad=0.3", "fc": "#EAF4FC",
                      "ec": _C0, "alpha": 0.85})

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("ROC AUC")
    ax.set_title(f"Learning Curve — {model_name}", pad=12)
    ax.set_ylim(bottom=max(0, min(va_mean) - 0.1))
    ax.legend(fontsize=10)
    _clean_spines(ax)
    _save(fig, f"10_learning_curve_{model_name.replace(' ', '_')}.png")


def plot_calibration_curve(y_true, y_prob, model_name: str) -> None:
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Reliability zones
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.06, color=_C1,
                    label="Over-confident zone")
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.06, color=_C0,
                    label="Under-confident zone")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", color=_C0, linewidth=2.2,
            markersize=7, markeredgecolor="white", markeredgewidth=1.2,
            label=model_name)

    # Histogram of predicted probs
    ax2 = ax.twinx()
    ax2.hist(y_prob, bins=20, alpha=0.15, color=_NAVY,
             edgecolor="none", density=True)
    ax2.set_ylabel("Predicted probability density", color=_NAVY, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=_NAVY, labelsize=8)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 4)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#D0D8E0")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve — {model_name}", pad=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9)
    _clean_spines(ax)
    _save(fig, f"11_calibration_{model_name.replace(' ', '_')}.png")
