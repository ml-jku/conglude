import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    from sklearn.metrics import auc, roc_curve
except Exception:
    auc = None
    roc_curve = None


METRIC_DISPLAY_NAMES = {
    "auc": "AUC-ROC",
    "auprc": "AUC-PR",
    "delta_auprc": "ΔAUC-PR",
    "bedroc": "BEDROC",
    "ef_0.005": "EF 0.5%",
    "ef_0.01": "EF 1%",
    "ef_0.05": "EF 5%",
    "dcc": "DCC",
    "dcc_ranked": "DCC (ranked)",
    "dca": "DCA",
    "dca_ranked": "DCA (ranked)",
    "iou": "IoU",
    "dcc_rank": "DCC (top-rank)",
    "dcc_conf": "DCC (top-conf)",
    "avg_num_pockets": "Avg. Pockets",
}


def _display_name(metric: str) -> str:
    return METRIC_DISPLAY_NAMES.get(metric, metric)


@dataclass(frozen=True)
class ResultsLayout:
    """
    Resolved file-system locations for a single evaluation run.

    Parameters
    ----------
    run_dir : Path
        Root directory of the timestamped results run.
    predictions_dir : Path
        Directory containing prediction CSV files.
    metrics_dir : Path
        Directory containing metrics CSV files.
    plots_dir : Path
        Directory where generated plot files are saved.
    """

    run_dir: Path
    predictions_dir: Path
    metrics_dir: Path
    plots_dir: Path



def resolve_run_dir(
    path: str
) -> Path:
    """
    Resolve a path to the root of a results run directory.

    If `path` points to a file or subdirectory inside a run, walks up the
    directory tree to find the run root.

    Parameters
    ----------
    path : str
        Path to the run directory or any file/subdirectory inside it.

    Returns
    -------
    Path
        The resolved run directory.
    """

    path = Path(path)
    if path.is_dir():
        return path

    if path.parent.name in {"predictions", "metrics", "embeddings", "plots"}:
        return path.parent.parent

    if path.parent.parent.name in {"predictions", "metrics", "embeddings", "plots"}:
        return path.parent.parent.parent

    return path.parent


def resolve_layout(
    path: str
) -> ResultsLayout:
    """
    Create a ResultsLayout from a run directory path.

    Only creates the plots directory; predictions and metrics directories
    are expected to already exist from the evaluation run.

    Parameters
    ----------
    path : str
        Path to the run directory or any file/subdirectory inside it.

    Returns
    -------
    ResultsLayout
        Resolved layout with all directory paths.
    """

    run_dir = resolve_run_dir(path)
    layout = ResultsLayout(
        run_dir=run_dir,
        predictions_dir=run_dir / "predictions",
        metrics_dir=run_dir / "metrics",
        plots_dir=run_dir / "plots",
    )
    layout.plots_dir.mkdir(parents=True, exist_ok=True)
    return layout


def _protein_sort_key(
    name: str
) -> Tuple[str, int, str]:
    """
    Generate a sort key for protein names that orders by prefix
    alphabetically then by numeric suffix.

    Parameters
    ----------
    name : str
        Protein name string (e.g. "3KYT", "5VB3_1").

    Returns
    -------
    Tuple[str, int, str]
        Sort key tuple of (prefix, numeric_suffix, full_name).
    """

    prefix = name.split("_")[0]
    suffix = name[len(prefix):]
    digits = "".join(ch for ch in suffix if ch.isdigit())
    return prefix, int(digits) if digits else -1, name



def load_vs_predictions(
    predictions_dir: str
) -> pd.DataFrame:
    """
    Load virtual screening predictions from CSV.

    Parameters
    ----------
    predictions_dir : str
        Directory containing vs_predictions.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: protein_name, ligand_idx, vs_pred, vs_label.
    """

    path = Path(predictions_dir) / "vs_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing VS predictions file: {path}")
    return pd.read_csv(path)


def load_pp_predictions(
    predictions_dir: str
) -> pd.DataFrame:
    """
    Load pocket prediction results from CSV.

    Parameters
    ----------
    predictions_dir : str
        Directory containing pp_predictions.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: protein_name, pocket_name, pred_x, pred_y, pred_z, confidence.
    """

    path = Path(predictions_dir) / "pp_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing pocket predictions file: {path}")
    return pd.read_csv(path)


def load_metrics(
    metrics_dir: str
) -> pd.DataFrame:
    """
    Load per-protein evaluation metrics from CSV.

    Parameters
    ----------
    metrics_dir : str
        Directory containing metrics.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with protein names and metric columns (auc, bedroc, ef_*).
    """

    path = Path(metrics_dir) / "metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return pd.read_csv(path)


def save_figure(
    path: str
) -> None:
    """
    Save the current matplotlib figure to disk and close it.

    Parameters
    ----------
    path : str
        Output file path for the figure.
    """

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_distributions(
    vs_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Plot per-protein score distributions for actives vs inactives.

    Generates one subplot per protein showing kernel density estimates
    (if seaborn is available) or histograms of prediction scores.

    Parameters
    ----------
    vs_df : pd.DataFrame
        Virtual screening predictions with columns: protein_name, vs_pred, vs_label.
    output_path : str
        Path to save the output PNG.
    """

    proteins = sorted(vs_df["protein_name"].dropna().unique(), key=_protein_sort_key)
    n_cols = min(4, max(1, len(proteins)))
    n_rows = int(np.ceil(len(proteins) / n_cols)) if proteins else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

    for idx, protein in enumerate(proteins):
        ax = axes[idx // n_cols, idx % n_cols]
        subset = vs_df[vs_df["protein_name"] == protein]
        actives = subset[subset["vs_label"] == 1]["vs_pred"]
        inactives = subset[subset["vs_label"] == 0]["vs_pred"]

        if sns is not None and len(actives) > 1 and len(inactives) > 1:
            sns.kdeplot(inactives, ax=ax, label="Inactives", color="steelblue", fill=True, alpha=0.35)
            sns.kdeplot(actives, ax=ax, label="Actives", color="crimson", fill=True, alpha=0.35)
        else:
            ax.hist(inactives, bins=50, alpha=0.6, label="Inactives", density=True, color="steelblue")
            ax.hist(actives, bins=50, alpha=0.6, label="Actives", density=True, color="crimson")

        ax.set_title(protein, fontsize=9)
        ax.legend(fontsize=7)

    for idx in range(len(proteins), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    plt.suptitle("Score Distributions: Actives vs Inactives", fontsize=12)
    plt.tight_layout()
    save_figure(output_path)


def plot_roc_curves(
    vs_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Plot per-protein ROC curves with a mean curve overlay.

    Parameters
    ----------
    vs_df : pd.DataFrame
        Virtual screening predictions with columns: protein_name, vs_pred, vs_label.
    output_path : str
        Path to save the output PNG.
    """

    if roc_curve is None or auc is None:
        raise RuntimeError("scikit-learn is required for ROC curve plotting.")

    proteins = sorted(vs_df["protein_name"].dropna().unique(), key=_protein_sort_key)
    fig, ax = plt.subplots(figsize=(6, 6))
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []

    for protein in proteins:
        subset = vs_df[vs_df["protein_name"] == protein]
        if subset["vs_label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(subset["vs_label"], subset["vs_pred"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, alpha=0.25, linewidth=0.8, label=f"{protein} ({roc_auc:.2f})")
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, color="black", linewidth=2, label=f"Mean (AUC={mean_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    save_figure(output_path)


def _enrichment_curve(
    subset: pd.DataFrame
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute the enrichment curve for a single protein.

    Parameters
    ----------
    subset : pd.DataFrame
        Predictions for a single protein with columns: vs_pred, vs_label.

    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray]]
        Tuple of (fraction_screened, fraction_actives_recovered), or None if no actives exist.
    """

    ordered = subset.sort_values("vs_pred", ascending=False)
    n_actives_total = ordered["vs_label"].sum()
    if n_actives_total == 0:
        return None
    cumulative_actives = ordered["vs_label"].cumsum().to_numpy() / n_actives_total
    fraction_screened = np.arange(1, len(ordered) + 1) / len(ordered)
    return fraction_screened, cumulative_actives


def plot_enrichment_curves(
    vs_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Plot per-protein enrichment curves with a mean curve overlay.

    Parameters
    ----------
    vs_df : pd.DataFrame
        Virtual screening predictions with columns: protein_name, vs_pred, vs_label.
    output_path : str
        Path to save the output PNG.
    """

    proteins = sorted(vs_df["protein_name"].dropna().unique(), key=_protein_sort_key)
    fig, ax = plt.subplots(figsize=(6, 5))
    mean_curves = []

    for protein in proteins:
        subset = vs_df[vs_df["protein_name"] == protein]
        curve = _enrichment_curve(subset)
        if curve is None:
            continue
        x, y = curve
        ax.plot(x, y, alpha=0.25, linewidth=0.8)
        mean_curves.append(np.interp(np.linspace(0, 1, 200), x, y))

    if mean_curves:
        mean_y = np.mean(mean_curves, axis=0)
        mean_x = np.linspace(0, 1, 200)
        ax.plot(mean_x, mean_y, color="black", linewidth=2, label="Mean")

    ax.set_xlabel("Fraction of library screened")
    ax.set_ylabel("Fraction of actives recovered")
    ax.set_title("Enrichment Curves")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    save_figure(output_path)


def plot_metric_comparison(
    metrics_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Plot per-metric bar charts comparing protein targets.

    Generates a combined figure with one subplot per metric, and also saves
    individual per-metric figures as metric_<name>.png in the same directory.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Per-protein metrics with a protein identifier column and one or more metric columns.
    output_path : str
        Path to save the combined metric comparison PNG.
    """

    id_columns = {"protein", "protein_name", "molecule"}
    metric_columns = [col for col in metrics_df.columns if col.lower() not in id_columns]
    if not metric_columns:
        raise ValueError("No metric columns found in metrics.csv")

    display_df = metrics_df.copy()
    if "molecule" in display_df.columns:
        id_column = "molecule"
    elif "protein" in display_df.columns:
        id_column = "protein"
    else:
        id_column = "protein_name"
    display_df = display_df.sort_values(id_column, key=lambda s: s.map(lambda x: _protein_sort_key(str(x))))

    output_path = Path(output_path)
    output_dir = output_path.parent

    n_metrics = len(metric_columns)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), squeeze=False)

    for idx, metric in enumerate(metric_columns):
        ax = axes[0, idx]
        x = np.arange(len(display_df))
        ax.bar(x, display_df[metric], color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(display_df[id_column], rotation=90, fontsize=7)
        ax.set_ylabel(_display_name(metric))
        ax.set_title(_display_name(metric))

    plt.suptitle("Metric Comparison", fontsize=12)
    plt.tight_layout()
    save_figure(output_path)

    for metric in metric_columns:
        fig, ax = plt.subplots(figsize=(max(4, len(display_df) * 0.5), 4))
        x = np.arange(len(display_df))
        ax.bar(x, display_df[metric], color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(display_df[id_column], rotation=90, fontsize=7)
        ax.set_ylabel(_display_name(metric))
        ax.set_title(_display_name(metric))
        plt.tight_layout()
        save_figure(output_dir / f"metric_{metric}.png")


def load_summary(
    metrics_dir: str
) -> pd.DataFrame:
    """
    Load dataset-wide aggregate metrics from summary CSV.

    Parameters
    ----------
    metrics_dir : str
        Directory containing summary.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: metric, value.
    """

    path = Path(metrics_dir) / "summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary file: {path}")
    return pd.read_csv(path)


def plot_summary(
    summary_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Plot a bar chart of dataset-wide aggregate metrics.

    Generates a single figure with one bar per metric, annotated with
    the numeric value above each bar. For PR tasks, dcc_rank and dcc_conf
    bars include Wilson confidence interval error bars.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Aggregate metrics with columns: metric, value.
    output_path : str
        Path to save the output PNG.
    """

    output_path = Path(output_path)

    # Build a lookup from short metric name to value for CI resolution
    metric_lookup = {}
    for _, row in summary_df.iterrows():
        short = row["metric"].split("/", 1)[-1]
        metric_lookup[short] = float(row["value"])

    # CI mapping: metric -> (ci_lower_key, ci_upper_key)
    ci_map = {
        "dcc_rank": ("ci_lower_rank", "ci_upper_rank"),
        "dcc_conf": ("ci_lower_conf", "ci_upper_conf"),
    }

    # Exclude CI rows from the plotted bars
    ci_keys = {"ci_lower_rank", "ci_upper_rank", "ci_lower_conf", "ci_upper_conf"}
    plot_df = summary_df[
        ~summary_df["metric"].apply(lambda m: m.split("/", 1)[-1]).isin(ci_keys)
    ].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(max(4, len(plot_df) * 0.8), 4))

    x = np.arange(len(plot_df))
    raw_labels = plot_df["metric"].apply(lambda m: m.split("/", 1)[-1])
    labels = [_display_name(l) for l in raw_labels]
    values = plot_df["value"].astype(float)

    # Compute error bars where CI data is available
    yerr_lower = np.zeros(len(plot_df))
    yerr_upper = np.zeros(len(plot_df))
    for i, raw_label in enumerate(raw_labels):
        if raw_label in ci_map:
            ci_lo_key, ci_hi_key = ci_map[raw_label]
            if ci_lo_key in metric_lookup and ci_hi_key in metric_lookup:
                yerr_lower[i] = values.iloc[i] - metric_lookup[ci_lo_key]
                yerr_upper[i] = metric_lookup[ci_hi_key] - values.iloc[i]

    has_errorbars = (yerr_lower.any() or yerr_upper.any())
    yerr = [yerr_lower, yerr_upper] if has_errorbars else None

    bars = ax.bar(x, values, color="steelblue", alpha=0.8,
                  yerr=yerr, capsize=4, ecolor="black")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Value")
    ax.set_title("Dataset Metrics Summary")
    plt.tight_layout()
    save_figure(output_path)


def is_target_fishing(
    layout: "ResultsLayout"
) -> bool:
    """
    Detect whether a results directory contains target fishing metrics.

    Checks the header of metrics.csv for a ``molecule`` column, which is
    present only in target fishing evaluations (VS/PP/PR use ``protein``).

    Parameters
    ----------
    layout : ResultsLayout
        Resolved directory layout for the evaluation run.

    Returns
    -------
    bool
        True if the dataset is a target fishing evaluation.
    """

    metrics_file = layout.metrics_dir / "metrics.csv"
    if not metrics_file.is_file():
        return False
    header = pd.read_csv(metrics_file, nrows=0).columns
    return "molecule" in header


def plot_tf_roc_curves(
    vs_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Plot per-molecule ROC curves for target fishing datasets.

    Groups predictions by ligand_idx (molecule) and plots one ROC curve per
    molecule with a mean curve overlay.

    Parameters
    ----------
    vs_df : pd.DataFrame
        TF predictions with columns: protein_name, ligand_idx, vs_pred, vs_label.
    output_path : str
        Path to save the output PNG.
    """

    if roc_curve is None or auc is None:
        raise RuntimeError("scikit-learn is required for ROC curve plotting.")

    molecules = sorted(vs_df["ligand_idx"].dropna().unique())
    fig, ax = plt.subplots(figsize=(6, 6))
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []

    for mol_idx in molecules:
        subset = vs_df[vs_df["ligand_idx"] == mol_idx]
        if subset["vs_label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(subset["vs_label"], subset["vs_pred"])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        ax.plot(fpr, tpr, alpha=0.05, linewidth=0.5, color="steelblue")

    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, color="crimson", linewidth=2, label=f"Mean (AUC={mean_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Target Fishing ROC Curves ({len(tprs)} molecules)")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    save_figure(output_path)


def plot_tf_enrichment_curves(
    vs_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Plot per-molecule enrichment curves for target fishing datasets.

    Groups predictions by ligand_idx (molecule) and plots fraction of active
    targets recovered vs fraction of target library screened.

    Parameters
    ----------
    vs_df : pd.DataFrame
        TF predictions with columns: protein_name, ligand_idx, vs_pred, vs_label.
    output_path : str
        Path to save the output PNG.
    """

    molecules = sorted(vs_df["ligand_idx"].dropna().unique())
    fig, ax = plt.subplots(figsize=(6, 5))
    mean_curves = []

    for mol_idx in molecules:
        subset = vs_df[vs_df["ligand_idx"] == mol_idx]
        curve = _enrichment_curve(subset)
        if curve is None:
            continue
        x, y = curve
        ax.plot(x, y, alpha=0.05, linewidth=0.5, color="steelblue")
        mean_curves.append(np.interp(np.linspace(0, 1, 200), x, y))

    if mean_curves:
        mean_y = np.mean(mean_curves, axis=0)
        mean_x = np.linspace(0, 1, 200)
        ax.plot(mean_x, mean_y, color="crimson", linewidth=2, label="Mean")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, label="Random")
    ax.set_xlabel("Fraction of target library screened")
    ax.set_ylabel("Fraction of active targets recovered")
    ax.set_title(f"Target Fishing Enrichment Curves ({len(mean_curves)} molecules)")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    save_figure(output_path)


def find_cleaned_pdb(
    dataset_root: str,
    protein_name: str
) -> Optional[Path]:
    """
    Locate the cleaned PDB file for a given protein.

    Parameters
    ----------
    dataset_root : str
        Root directory of the dataset (containing processed/cleaned_pdbs/).
    protein_name : str
        Protein identifier matching the subdirectory name under cleaned_pdbs/.

    Returns
    -------
    Optional[Path]
        Path to the cleaned PDB file, or None if not found.
    """

    dataset_root = Path(dataset_root)
    candidate = dataset_root / "processed" / "cleaned_pdbs" / protein_name / "protein.pdb"
    return candidate if candidate.is_file() else None


def build_pymol_scene_script(
    protein_name: str,
    pocket_df: pd.DataFrame
) -> str:
    """
    Build a PyMOL script that loads a protein and overlays numbered pocket centers.

    Parameters
    ----------
    protein_name : str
        Protein identifier, used as the PyMOL object name.
    pocket_df : pd.DataFrame
        Pocket predictions for a single protein with columns: pred_x, pred_y, pred_z.

    Returns
    -------
    str
        PyMOL script content.
    """

    lines = [
        "reinitialize",
        "set bg_rgb, white",
        f"load protein.pdb, {protein_name}",
        "hide everything",
        f"show cartoon, {protein_name}",
        f"color gray70, {protein_name}",
        f"set cartoon_transparency, 0.1, {protein_name}",
        "set label_color, black",
        "set label_size, 18",
        "set sphere_scale, 0.8",
        "set sphere_quality, 2",
    ]

    pocket_object_names = []
    ordered_df = pocket_df.reset_index(drop=True)

    for pocket_number, row in enumerate(ordered_df.itertuples(index=False), start=1):
        object_name = f"{protein_name}_pocket_{pocket_number}"
        pocket_object_names.append(object_name)
        lines.extend([
            f'pseudoatom {object_name}, pos=[{row.pred_x:.3f}, {row.pred_y:.3f}, {row.pred_z:.3f}], name="{pocket_number}", resn=PKT, resi="{pocket_number}", chain=A',
            f"show spheres, {object_name}",
            f"color tv_red, {object_name}",
            f"label {object_name}, name",
        ])

    if pocket_object_names:
        lines.append(f"group pockets, {' '.join(pocket_object_names)}")

    lines.extend([
        f"orient {protein_name}",
        f"zoom {protein_name}, 10",
    ])

    return "\n".join(lines) + "\n"


def create_pymol_scene(
    protein_name: str,
    protein_pdb: str,
    pocket_df: pd.DataFrame,
    output_dir: str
) -> Path:
    """
    Generate a PyMOL scene showing predicted pocket locations on a protein structure.

    Copies the cleaned PDB file into the output directory and writes a .pml script
    that loads the protein, renders it as cartoon, and places colored spheres at
    each predicted pocket center.

    Parameters
    ----------
    protein_name : str
        Protein identifier.
    protein_pdb : str
        Path to the cleaned PDB file.
    pocket_df : pd.DataFrame
        Pocket predictions for this protein with columns: pred_x, pred_y, pred_z.
    output_dir : str
        Directory where the PDB copy and .pml script will be written.

    Returns
    -------
    Path
        Path to the generated .pml scene script.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(protein_pdb, output_dir / "protein.pdb")
    scene_script = build_pymol_scene_script(protein_name, pocket_df)

    scene_path = output_dir / "view.pml"
    scene_path.write_text(scene_script, encoding="utf-8")
    return scene_path
