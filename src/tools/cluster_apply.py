from __future__ import annotations

"""Execute file organization by speaker clusters."""

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

console = Console()
logger = logging.getLogger(__name__)


def load_namelist(namelist_path: Path) -> dict[str, Any]:
    """Load and validate a namelist JSON file."""
    if not namelist_path.exists():
        console.print(f"[red]Namelist not found: {namelist_path}[/red]")
        sys.exit(1)

    with open(namelist_path, "r") as f:
        data: dict[str, Any] = json.load(f)

    if "_meta" not in data or "mappings" not in data:
        console.print(f"[red]Invalid namelist format: {namelist_path}[/red]")
        console.print("[dim]Expected keys: '_meta' and 'mappings'[/dim]")
        sys.exit(1)

    if not data["mappings"]:
        console.print(f"[yellow]Namelist has no mappings: {namelist_path}[/yellow]")
        return data

    return data


def discover_namelists(namelist_dir: Path) -> list[Path]:
    """Find all *_namelist.json files in a directory."""
    if not namelist_dir.exists():
        console.print(f"[red]Directory not found: {namelist_dir}[/red]")
        sys.exit(1)

    namelists = sorted(namelist_dir.glob("*_namelist.json"))
    legacy = namelist_dir / "namelist.json"
    if legacy.exists() and legacy not in namelists:
        namelists.append(legacy)

    if not namelists:
        console.print(f"[red]No namelist files found in {namelist_dir}[/red]")
        sys.exit(1)

    return namelists


def plan_moves(mappings: dict[str, str]) -> tuple[list[tuple[Path, Path]], dict[str, int]]:
    """Plan file moves from mappings."""
    move_plan: list[tuple[Path, Path]] = []
    stats: dict[str, int] = defaultdict(int)
    skipped = 0

    for original_path_str, speaker_id in mappings.items():
        src = Path(original_path_str)
        if not src.exists():
            logger.warning("File not found, skipping: %s", src)
            skipped += 1
            continue

        if src.parent.name.startswith("speaker_"):
            logger.debug("Already organized, skipping: %s", src)
            skipped += 1
            continue

        dst = src.parent / speaker_id / src.name
        move_plan.append((src, dst))
        stats[speaker_id] += 1

    if skipped > 0:
        console.print(f"[yellow]Skipped {skipped} files (not found or already organized)[/yellow]")

    return move_plan, dict(stats)


def preview_moves(move_plan: list[tuple[Path, Path]], stats: dict[str, int]) -> None:
    """Print a preview of planned file moves."""
    if not move_plan:
        console.print("[yellow]No files to move.[/yellow]")
        return

    table = Table(title="Move Summary", show_header=True)
    table.add_column("Speaker", style="cyan")
    table.add_column("Files", style="green", justify="right")
    for speaker_id, count in sorted(stats.items()):
        table.add_row(speaker_id, str(count))
    table.add_row("[bold]Total[/bold]", f"[bold]{len(move_plan)}[/bold]")
    console.print(table)

    console.print("\n[bold]Example moves (first 10):[/bold]")
    tree = Tree("[dim]Planned moves[/dim]")
    for src, dst in move_plan[:10]:
        tree.add(f"{src.name} → [cyan]{dst.parent.name}[/cyan]/{dst.name}")
    if len(move_plan) > 10:
        tree.add(f"[dim]... and {len(move_plan) - 10} more[/dim]")
    console.print(tree)


def execute_moves(move_plan: list[tuple[Path, Path]], backup_path: Path) -> int:
    """Execute file moves and save backup manifest."""
    backup_manifest: dict[str, Any] = {
        "_meta": {
            "generated_by": "Start-Clustering.py",
            "timestamp": datetime.now().isoformat(),
            "total_files": len(move_plan),
        },
        "moves": {},
    }

    for src, dst in move_plan:
        backup_manifest["moves"][str(dst)] = str(src)

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with open(backup_path, "w") as f:
        json.dump(backup_manifest, f, indent=2)
    console.print(f"[green]Backup manifest saved:[/green] {backup_path}")

    moved = 0
    errors = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Moving files...", total=len(move_plan))
        for src, dst in move_plan:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                moved += 1
            except (OSError, shutil.Error) as e:
                logger.error("Failed to move %s → %s: %s", src, dst, e)
                errors += 1
            progress.advance(task)

    if errors > 0:
        console.print(f"[red]Failed to move {errors} files. Check logs for details.[/red]")
    return moved


def undo_moves(backup_path: Path, dry_run: bool = False) -> None:
    """Undo a previous file organization using backup manifest."""
    if not backup_path.exists():
        console.print(f"[red]Backup manifest not found: {backup_path}[/red]")
        sys.exit(1)

    with open(backup_path, "r") as f:
        manifest: dict[str, Any] = json.load(f)

    if "moves" not in manifest:
        console.print(f"[red]Invalid backup manifest: {backup_path}[/red]")
        sys.exit(1)

    moves: dict[str, str] = manifest["moves"]
    console.print(f"[bold]Undoing {len(moves)} file moves...[/bold]")

    if dry_run:
        console.print("[yellow]DRY RUN — no files will be moved[/yellow]")
        for current_path, original_path in list(moves.items())[:10]:
            console.print(f"  [cyan]{Path(current_path).name}[/cyan] → {original_path}")
        if len(moves) > 10:
            console.print(f"  [dim]... and {len(moves) - 10} more[/dim]")
        return

    restored = 0
    errors = 0
    empty_dirs: set[Path] = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Restoring files...", total=len(moves))
        for current_path_str, original_path_str in moves.items():
            current = Path(current_path_str)
            original = Path(original_path_str)
            try:
                if current.exists():
                    original.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(current), str(original))
                    empty_dirs.add(current.parent)
                    restored += 1
                elif original.exists():
                    restored += 1
                else:
                    logger.warning("File not found at either location: %s", current)
                    errors += 1
            except (OSError, shutil.Error) as e:
                logger.error("Failed to restore %s → %s: %s", current, original, e)
                errors += 1
            progress.advance(task)

    cleaned = 0
    for dir_path in empty_dirs:
        try:
            if dir_path.exists() and not any(dir_path.iterdir()):
                dir_path.rmdir()
                cleaned += 1
        except OSError:
            pass

    console.print(f"[green]Restored {restored} files, cleaned {cleaned} empty directories[/green]")
    if errors > 0:
        console.print(f"[red]Failed to restore {errors} files[/red]")


def process_namelist(namelist_path: Path, output_dir: Path, dry_run: bool = False) -> int:
    """Process a single namelist file."""
    console.print(f"\n[bold blue]Processing: {namelist_path.name}[/bold blue]")
    data = load_namelist(namelist_path)
    mappings = data.get("mappings", {})
    meta = data.get("_meta", {})

    if not mappings:
        console.print("[yellow]No mappings to process.[/yellow]")
        return 0

    console.print(f"[dim]Files: {meta.get('total_files', len(mappings))}, Speakers: {meta.get('num_speakers', '?')}[/dim]")
    move_plan, stats = plan_moves(mappings)
    if not move_plan:
        console.print("[yellow]No files need moving.[/yellow]")
        return 0

    preview_moves(move_plan, stats)
    if dry_run:
        console.print("\n[yellow]DRY RUN — no files were moved[/yellow]")
        return 0

    stem = namelist_path.stem.replace("_namelist", "")
    backup_path = output_dir / f"{stem}_backup_manifest.json"
    moved = execute_moves(move_plan, backup_path)
    console.print(f"[green]Moved {moved} files into speaker directories[/green]")
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize audio files into speaker directories based on cluster-Test.py results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Organize from a specific namelist
    python Start-Clustering.py --namelist cluster_output/positive_namelist.json

    # Organize all namelists in a directory
    python Start-Clustering.py --namelist-dir cluster_output

    # Preview changes without moving files
    python Start-Clustering.py --namelist cluster_output/positive_namelist.json --dry-run

    # Undo a previous organization
    python Start-Clustering.py --undo cluster_output/positive_backup_manifest.json

    # Undo with preview
    python Start-Clustering.py --undo cluster_output/positive_backup_manifest.json --dry-run
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--namelist", type=str, default=None, help="Path to a namelist JSON file from cluster-Test.py")
    group.add_argument(
        "--namelist-dir",
        type=str,
        default=None,
        help="Directory containing namelist JSON files (processes all *_namelist.json)",
    )
    group.add_argument(
        "--undo",
        type=str,
        default=None,
        help="Path to backup manifest JSON to undo a previous organization",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./cluster_output",
        help="Directory for backup manifests (default: ./cluster_output)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without moving files")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.undo is not None:
        console.print(
            Panel.fit(
                "[bold yellow]Undo Mode[/bold yellow]\n[dim]Restoring files to original locations[/dim]",
                title="Start-Clustering",
                border_style="yellow",
            )
        )
        undo_moves(Path(args.undo), dry_run=args.dry_run)
        return

    if args.dry_run:
        console.print(
            Panel.fit(
                "[bold yellow]DRY RUN — No files will be moved[/bold yellow]\n[dim]Review the plan, then run without --dry-run to execute[/dim]",
                title="Start-Clustering",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel.fit(
                "[bold red]⚠ This will MOVE files in your dataset[/bold red]\n[dim]A backup manifest will be created for undo[/dim]",
                title="Start-Clustering",
                border_style="red",
            )
        )

    namelists = [Path(args.namelist)] if args.namelist is not None else discover_namelists(Path(args.namelist_dir))
    console.print(f"[dim]Found {len(namelists)} namelist(s) to process[/dim]")

    total_moved = 0
    for namelist_path in namelists:
        moved = process_namelist(namelist_path, output_dir, dry_run=args.dry_run)
        total_moved += moved

    console.print("\n" + "=" * 70)
    if args.dry_run:
        console.print(
            Panel(
                "[bold yellow]Dry Run Complete[/bold yellow]\n\nNo files were moved. Review the plan above.\nRun without [bold]--dry-run[/bold] to execute.",
                title="Done",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold green]Organization Complete![/bold green]\n\n"
                f"Moved [cyan]{total_moved}[/cyan] files into speaker directories.\n\n"
                "To undo, run:\n"
                f"  [bold]python Start-Clustering.py --undo {output_dir}/<dataset>_backup_manifest.json[/bold]",
                title="Done",
                border_style="green",
            )
        )
