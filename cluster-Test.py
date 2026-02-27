#!/usr/bin/env python3
"""
cluster-Test.py - Speaker Clustering Dry-Run Analysis Tool

Analyzes speaker clusters in dataset directories without moving files.
Supports positive, negative, and hard_negative datasets.
Generates a detailed report and creates namelist.json for review.

Usage:
    mww-torch
    python cluster-Test.py --config standard
    python cluster-Test.py --config standard --dataset all
    python cluster-Test.py --config standard --dataset negative
    python cluster-Test.py --config standard --override my_config.yaml

Output:
    - Console report with cluster statistics
    - {dataset}_namelist.json mapping file for inspection
    - {dataset}_cluster_report.txt per dataset

This script is READ-ONLY on your dataset. No files are moved.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore", category=UserWarning)

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.loader import ConfigLoader, load_full_config
from src.data.clustering import SpeakerClustering, SpeakerClusteringConfig

console = Console()
logger = logging.getLogger(__name__)


def discover_audio_files(audio_dir: Path) -> List[Path]:
    """Discover all audio files in a directory recursively.

    Returns empty list if directory doesn't exist or has no audio files.
    """
    if not audio_dir.exists():
        console.print(f"[yellow]Directory not found, skipping: {audio_dir}[/yellow]")
        return []

    console.print(f"[dim]Scanning {audio_dir} for audio files...[/dim]")

    audio_files: List[Path] = []
    for ext in ["*.wav", "*.WAV", "*.mp3", "*.MP3", "*.flac", "*.FLAC"]:
        audio_files.extend(audio_dir.rglob(ext))

    # Sort for reproducibility
    audio_files = sorted(audio_files)

    if not audio_files:
        console.print(f"[yellow]No audio files found in {audio_dir}[/yellow]")
        return []

    console.print(f"[green]Found {len(audio_files)} audio files[/green]")
    return audio_files


def analyze_clusters(
    audio_paths: List[Path],
    clusterer: SpeakerClustering,
    config: SpeakerClusteringConfig,
) -> Tuple[Dict[str, Any], Dict[int, List[Path]]]:
    """
    Analyze clusters and return detailed statistics.

    Returns:
        Tuple of (analysis report dict, cluster mapping dict)
    """
    console.print("\n[bold blue]Extracting speaker embeddings...[/bold blue]")
    console.print(f"[dim]Model: {config.embedding_model}[/dim]")

    # Convert paths to strings for clustering
    path_strs = [str(p) for p in audio_paths]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Clustering samples...", total=1)

        # Get cluster assignments
        path_to_cluster = clusterer.cluster_samples(path_strs)
        progress.update(task, advance=1)

    # Build cluster mapping
    cluster_to_paths: Dict[int, List[Path]] = defaultdict(list)
    for path_str, cluster_id in path_to_cluster.items():
        cluster_to_paths[cluster_id].append(Path(path_str))

    # Calculate statistics
    total_files = len(audio_paths)
    num_clusters = len(cluster_to_paths)
    samples_per_cluster = [len(paths) for paths in cluster_to_paths.values()]

    analysis = {
        "total_files": total_files,
        "num_clusters": num_clusters,
        "samples_per_cluster": samples_per_cluster,
        "min_cluster_size": min(samples_per_cluster),
        "max_cluster_size": max(samples_per_cluster),
        "avg_cluster_size": sum(samples_per_cluster) / len(samples_per_cluster),
        "threshold_used": config.similarity_threshold,
        "embedding_model": config.embedding_model,
        "method": config.method,
        "clusters": {},
    }

    # Analyze each cluster
    for cluster_id, paths in sorted(cluster_to_paths.items()):
        # Extract directory info for this cluster
        dir_counts = defaultdict(int)
        for p in paths:
            dir_counts[str(p.parent)] += 1

        # Get sample filenames (first 5)
        sample_files = [p.name for p in paths[:5]]

        analysis["clusters"][f"cluster_{cluster_id}"] = {
            "size": len(paths),
            "percentage": len(paths) / total_files * 100,
            "directories": dict(dir_counts),
            "sample_files": sample_files,
        }

    return analysis, cluster_to_paths


def print_cluster_report(analysis: Dict[str, Any], dataset_label: str = "positive") -> None:
    """Print a formatted cluster analysis report."""
    console.print("\n" + "=" * 70)
    console.print(
        Panel.fit(f"[bold cyan]Speaker Clustering Analysis Report — {dataset_label}[/bold cyan]\n" "[dim]Dry-run mode - No files were moved[/dim]", title="cluster-Test", border_style="cyan")
    )
    console.print("=" * 70)

    # Summary table
    summary_table = Table(title="Summary Statistics", show_header=True)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Total Audio Files", str(analysis["total_files"]))
    summary_table.add_row("Number of Clusters", str(analysis["num_clusters"]))
    summary_table.add_row("Similarity Threshold", f"{analysis['threshold_used']:.2f}")
    summary_table.add_row("Embedding Model", analysis["embedding_model"])
    summary_table.add_row("Clustering Method", analysis["method"])
    summary_table.add_row("Min Cluster Size", str(analysis["min_cluster_size"]))
    summary_table.add_row("Max Cluster Size", str(analysis["max_cluster_size"]))
    summary_table.add_row("Avg Cluster Size", f"{analysis['avg_cluster_size']:.1f}")

    console.print(summary_table)

    # Cluster details
    console.print("\n[bold]Cluster Details:[/bold]")

    for cluster_name, cluster_info in sorted(analysis["clusters"].items()):
        cluster_id = int(cluster_name.replace("cluster_", ""))
        size = cluster_info["size"]
        pct = cluster_info["percentage"]

        # Create tree for this cluster
        tree = Tree(f"[bold]Cluster {cluster_id}[/bold] ([green]{size}[/green] files, [yellow]{pct:.1f}%[/yellow])")

        # Add directory breakdown
        dirs_branch = tree.add("[dim]Source directories:[/dim]")
        for dir_path, count in list(cluster_info["directories"].items())[:5]:
            dirs_branch.add(f"{dir_path} ([cyan]{count}[/cyan] files)")
        if len(cluster_info["directories"]) > 5:
            dirs_branch.add(f"[dim]... and {len(cluster_info['directories']) - 5} more[/dim]")

        # Add sample files
        samples_branch = tree.add("[dim]Sample files:[/dim]")
        for fname in cluster_info["sample_files"][:3]:
            samples_branch.add(f"[dim]{fname}[/dim]")
        if len(cluster_info["sample_files"]) > 3:
            samples_branch.add(f"[dim]... and {size - 3} more[/dim]")

        console.print(tree)
        console.print()


def save_namelist_json(
    output_path: Path,
    cluster_to_paths: Dict[int, List[Path]],
    analysis: Dict[str, Any],
) -> None:
    """Save the cluster mapping to namelist.json."""

    # Build the mapping: relative_path -> speaker_id
    namelist = {
        "_meta": {
            "generated_by": "cluster-Test.py",
            "version": "1.0",
            "total_files": analysis["total_files"],
            "num_speakers": analysis["num_clusters"],
            "similarity_threshold": analysis["threshold_used"],
            "embedding_model": analysis["embedding_model"],
            "method": analysis["method"],
        },
        "mappings": {},
    }

    for cluster_id, paths in sorted(cluster_to_paths.items()):
        speaker_id = f"speaker_{cluster_id:04d}"
        for path in paths:
            # Store relative path from dataset root
            rel_path = str(path)
            namelist["mappings"][rel_path] = speaker_id

    with open(output_path, "w") as f:
        json.dump(namelist, f, indent=2, sort_keys=True)

    console.print(f"[green]Saved namelist to:[/green] {output_path}")
    console.print(f"[dim]This file maps {len(namelist['mappings'])} files to {analysis['num_clusters']} speakers[/dim]")


def save_cluster_report(output_path: Path, analysis: Dict[str, Any]) -> None:
    """Save a detailed text report."""
    lines = [
        "Speaker Clustering Analysis Report",
        "=" * 70,
        "",
        "SUMMARY",
        "-" * 70,
        f"Total Audio Files:    {analysis['total_files']}",
        f"Number of Clusters:   {analysis['num_clusters']}",
        f"Similarity Threshold: {analysis['threshold_used']:.2f}",
        f"Embedding Model:      {analysis['embedding_model']}",
        f"Clustering Method:    {analysis['method']}",
        "",
        "CLUSTER STATISTICS",
        "-" * 70,
        f"Min Cluster Size:     {analysis['min_cluster_size']}",
        f"Max Cluster Size:     {analysis['max_cluster_size']}",
        f"Avg Cluster Size:     {analysis['avg_cluster_size']:.1f}",
        "",
        "CLUSTER DETAILS",
        "-" * 70,
        "",
    ]

    for cluster_name, info in sorted(analysis["clusters"].items()):
        cluster_id = int(cluster_name.replace("cluster_", ""))
        lines.append(f"Cluster {cluster_id}: {info['size']} files ({info['percentage']:.1f}%)")
        lines.append("  Directories:")
        for dir_path, count in list(info["directories"].items())[:5]:
            lines.append(f"    - {dir_path}: {count} files")
        if len(info["directories"]) > 5:
            lines.append(f"    ... and {len(info['directories']) - 5} more directories")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    console.print(f"[green]Saved report to:[/green] {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze speaker clusters without moving files (DRY-RUN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cluster-Test.py --config standard
    python cluster-Test.py --config standard --dataset all
    #PR|    python cluster-Test.py --config standard --n-clusters 200
#TK|    python cluster-Test.py --config standard --override my_config.yaml

Output files (per dataset):
    python cluster-Test.py --config standard --n-clusters 200
    #TK|    python cluster-Test.py --config standard --override my_config.yaml

#NN|Output files (per dataset):

Output files (per dataset):
Output files (per dataset):
    - {dataset}_namelist.json: Mapping of file paths to speaker IDs
    - {dataset}_cluster_report.txt: Human-readable report
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config preset name (standard, fast_test, max_quality) or path to YAML file",
    )

    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Override config file (optional)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./cluster_output",
        help="Directory for output files (default: ./cluster_output)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override similarity threshold (default: from config)",
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Explicit number of clusters (overrides threshold). Use when you know approximate speaker count.",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="positive",
        choices=["positive", "negative", "hard_negative", "all"],
        help="Which dataset(s) to cluster (default: positive)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Load configuration
    console.print("[bold]Loading configuration...[/bold]")

    if args.config in ConfigLoader.VALID_PRESETS:
        config = load_full_config(args.config, args.override)
    else:
        config_path = Path(args.config)
        if not config_path.exists():
            console.print(f"[red]Config file not found: {args.config}[/red]")
            sys.exit(1)
        config = load_full_config(None, str(config_path))

    # Get clustering config
    cluster_config = config.speaker_clustering

    # Override threshold if specified
    if args.threshold is not None:
        cluster_config.similarity_threshold = args.threshold
        console.print(f"[yellow]Using custom threshold: {args.threshold}[/yellow]")

    # Override n_clusters if specified (takes priority over threshold)
    if args.n_clusters is not None:
        cluster_config.n_clusters = args.n_clusters
        console.print(f"[yellow]Using explicit cluster count: {args.n_clusters} (overrides threshold)[/yellow]")

    # Determine which datasets to process
    dataset_dirs: Dict[str, Path] = {
        "positive": Path(config.paths.positive_dir),
        "negative": Path(config.paths.negative_dir),
        "hard_negative": Path(config.paths.hard_negative_dir),
    }

    if args.dataset == "all":
        datasets_to_run = ["positive", "negative", "hard_negative"]
    else:
        datasets_to_run = [args.dataset]

    console.print("[green]Config loaded successfully[/green]")
    console.print(f"[dim]Datasets to cluster: {', '.join(datasets_to_run)}[/dim]")
    if cluster_config.n_clusters is not None:
        console.print(f"[dim]Cluster count: {cluster_config.n_clusters} (explicit)[/dim]")
    else:
        console.print(f"[dim]Similarity threshold: {cluster_config.similarity_threshold}[/dim]")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clusterer = SpeakerClustering(cluster_config)
    all_reports: List[Path] = []
    all_namelists: List[Path] = []

    for dataset_name in datasets_to_run:
        dataset_dir = dataset_dirs[dataset_name]
        console.print(f"\n[bold magenta]{'=' * 70}[/bold magenta]")
        console.print(f"[bold magenta]  Dataset: {dataset_name} ({dataset_dir})[/bold magenta]")
        console.print(f"[bold magenta]{'=' * 70}[/bold magenta]")

        # Discover audio files
        audio_files = discover_audio_files(dataset_dir)

        if not audio_files:
            console.print(f"[yellow]Skipping {dataset_name} — no audio files found[/yellow]")
            continue

        if args.max_files:
            audio_files = audio_files[: args.max_files]
            console.print(f"[yellow]Limited to {len(audio_files)} files[/yellow]")

        # Run clustering analysis
        console.print(f"\n[bold blue]Running clustering analysis for {dataset_name}...[/bold blue]")
        analysis, cluster_to_paths = analyze_clusters(audio_files, clusterer, cluster_config)

        # Print report
        print_cluster_report(analysis, dataset_label=dataset_name)

        # Save outputs with dataset prefix
        console.print(f"\n[bold]Saving output files for {dataset_name}...[/bold]")

        namelist_path = output_dir / f"{dataset_name}_namelist.json"
        save_namelist_json(namelist_path, cluster_to_paths, analysis)
        all_namelists.append(namelist_path)

        report_path = output_dir / f"{dataset_name}_cluster_report.txt"
        save_cluster_report(report_path, analysis)
        all_reports.append(report_path)

    # Final summary
    if not all_reports:
        console.print("\n[red]No datasets had audio files to cluster.[/red]")
        sys.exit(1)

    console.print("\n" + "=" * 70)
    report_lines = "\n".join(f"  - [cyan]{r}[/cyan]" for r in all_reports)
    namelist_lines = "\n".join(f"  - [cyan]{n}[/cyan]" for n in all_namelists)
    console.print(
        Panel(
            f"[bold green]Analysis Complete! ({len(all_reports)} dataset(s) processed)[/bold green]\n\n"
            "Generated reports:\n"
            f"{report_lines}\n\n"
            "Generated namelists:\n"
            f"{namelist_lines}\n\n"
            "Next steps:\n"
            "1. Review the reports and namelists above\n"
            "2. Inspect sample files in each cluster\n"
            "3. Run Start-Clustering.py to organize files into speaker directories:\n"
            "   python Start-Clustering.py --namelist-dir cluster_output --dry-run\n"
            "   python Start-Clustering.py --namelist-dir cluster_output",
            title="Done",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
