import argparse
from pathlib import Path

from conglude.utils.visualization import (
    ResultsLayout,
    resolve_layout,
    is_target_fishing,
    load_vs_predictions,
    load_pp_predictions,
    load_metrics,
    load_summary,
    plot_score_distributions,
    plot_roc_curves,
    plot_enrichment_curves,
    plot_metric_comparison,
    plot_summary,
    plot_tf_roc_curves,
    plot_tf_enrichment_curves,
    find_cleaned_pdb,
    create_pymol_scene,
    _protein_sort_key,
)



def generate_all(
    layout: ResultsLayout,
    include_pymol: bool = True,
    dataset_root: str = None,
) -> None:
    """
    Generate all visualization outputs for an evaluation run.

    Produces task-appropriate plots depending on the dataset type:
    - VS: score distributions, ROC curves, enrichment curves, per-target metric bar charts
    - TF: per-molecule ROC curves, per-molecule enrichment curves
    - PP/PR: aggregate metrics summary bar chart
    - All tasks: dataset-wide metrics summary bar chart

    Optionally generates PyMOL scene files for predicted pockets.
    Gracefully skips plots when the required files are missing.

    Parameters
    ----------
    layout : ResultsLayout
        Resolved directory layout for the evaluation run.
    include_pymol : bool
        Whether to generate PyMOL scene files from pocket predictions.
    dataset_root : str, optional
        Root directory of the dataset, used to locate cleaned PDB files for
        PyMOL scenes. If None, inferred from the run directory structure.
    """

    tf_dataset = is_target_fishing(layout)

    vs_path = layout.predictions_dir / "vs_predictions.csv"
    if vs_path.is_file():
        vs_df = load_vs_predictions(layout.predictions_dir)
        if tf_dataset:
            plot_tf_roc_curves(vs_df, layout.plots_dir / "roc_curves.png")
            plot_tf_enrichment_curves(vs_df, layout.plots_dir / "enrichment_curves.png")
        else:
            plot_score_distributions(vs_df, layout.plots_dir / "score_distributions.png")
            plot_roc_curves(vs_df, layout.plots_dir / "roc_curves.png")
            plot_enrichment_curves(vs_df, layout.plots_dir / "enrichment_curves.png")

    metrics_file = layout.metrics_dir / "metrics.csv"
    if metrics_file.is_file() and not tf_dataset:
        metrics_df = load_metrics(layout.metrics_dir)
        plot_metric_comparison(metrics_df, layout.plots_dir / "metric_comparison.png")

    summary_file = layout.metrics_dir / "summary.csv"
    if summary_file.is_file():
        summary_df = load_summary(layout.metrics_dir)
        plot_summary(summary_df, layout.plots_dir / "summary.png")

    if include_pymol:
        pp_path = layout.predictions_dir / "pp_predictions.csv"
        if pp_path.is_file():
            pp_df = load_pp_predictions(layout.predictions_dir)
            resolved_dataset_root = Path(dataset_root) if dataset_root is not None else (
                layout.run_dir.parent.parent if layout.run_dir.parent.name == "results" else None
            )
            if resolved_dataset_root is not None:
                protein_names = sorted(pp_df["protein_name"].dropna().unique(), key=_protein_sort_key)
                for protein_name in protein_names:
                    protein_pdb = find_cleaned_pdb(resolved_dataset_root, protein_name)
                    if protein_pdb is None:
                        continue
                    pocket_df = pp_df[pp_df["protein_name"] == protein_name].reset_index(drop=True)
                    create_pymol_scene(
                        protein_name=protein_name,
                        protein_pdb=protein_pdb,
                        pocket_df=pocket_df,
                        output_dir=layout.plots_dir / protein_name,
                    )



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate ConGLUDe evaluation visualizations.")
    parser.add_argument("--run-dir", nargs="+", required=True,
                        help="Path(s) to timestamped results run directories. Multiple directories are processed sequentially.")
    parser.add_argument("--dataset-root", default=None, help="Optional dataset root used to locate cleaned PDBs for PyMOL scenes.")
    parser.add_argument("--no-pymol", action="store_true", help="Skip PyMOL scene generation.")
    args = parser.parse_args()

    for run_dir in args.run_dir:
        print(f"Processing: {run_dir}")
        layout = resolve_layout(run_dir)
        generate_all(layout, include_pymol=not args.no_pymol, dataset_root=args.dataset_root)
