#!/usr/bin/env python3
"""
Tidy and format YAML configuration files.

This script:
1. Sorts keys alphabetically within each section
2. Fixes indentation (standardizes to 2 spaces)
3. Removes trailing whitespace
4. Ensures consistent formatting
5. Adds proper spacing between sections

Usage:
    python scripts/tidy_yaml.py --config-dir config/ --dry-run  # Preview
    python scripts/tidy_yaml.py --config-dir config/ --apply    # Apply changes
"""

import argparse
import re
import sys
from pathlib import Path
from collections import OrderedDict
from typing import List, Dict, Any

if __name__ == "__main__":
    sys.exit(main())


def parse_yaml_simple(content: str) -> OrderedDict:
    """
    Simple YAML parser that preserves structure and comments.
    Returns an OrderedDict of sections.
    """
    lines = content.split("\n")
    sections = OrderedDict()
    current_section = "__preamble__"
    current_content = []

    for line in lines:
        # Check for section header (top-level key)
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*:\s*$", line):
            # Save previous section (including preamble)
            sections[current_section] = current_content

            # Start new section
            current_section = line.rstrip(":").strip()
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_content or current_section == "__preamble__":
        sections[current_section] = current_content

    return sections


def sort_section_keys(lines: List[str]) -> List[str]:
    """Sort keys within a section while preserving structure."""
    if not lines:
        return lines

    # Parse key-value pairs
    entries = []
    current_entry = []
    current_key = None

    for line in lines:
        stripped = line.strip()

        # Check if this is a new key (2-space indent followed by key:)
        if re.match(r"^  [a-zA-Z_][a-zA-Z0-9_]*:", line) or re.match(r"^  -", line):
            # Save previous entry
            if current_key is not None:
                entries.append((current_key, current_entry))

            # Start new entry
            if ":" in stripped:
                current_key = stripped.split(":")[0].strip()
            else:
                current_key = stripped
            current_entry = [line]
        elif current_key is not None:
            current_entry.append(line)

    # Save last entry
    if current_key is not None:
        entries.append((current_key, current_entry))
    else:
        # Just return original lines if no keys found
        return lines

    # Sort entries by key
    entries.sort(key=lambda x: x[0].lower())

    # Reconstruct lines
    result = []
    for i, (key, entry_lines) in enumerate(entries):
        result.extend(entry_lines)

    return result


def remove_trailing_whitespace(lines: List[str]) -> List[str]:
    """Remove trailing whitespace from lines."""
    return [line.rstrip() for line in lines]


def fix_indentation(lines: List[str]) -> List[str]:
    """Fix indentation to use consistent 2-space indents."""
    result = []
    for line in lines:
        if "\t" in line:
            # Replace tabs with spaces
            line = line.replace("\t", "  ")
        result.append(line)
    return result


def process_yaml_file(filepath: Path, dry_run: bool = True) -> Dict[str, Any]:
    """Process a single YAML file."""
    changes = {"trailing_ws": 0, "tabs": 0, "sorted": [], "modified": False}

    try:
        with open(filepath, "r") as f:
            content = f.read()
            original_lines = content.split("\n")
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return changes

    # Remove trailing whitespace
    lines = remove_trailing_whitespace(original_lines)
    changes["trailing_ws"] = sum(1 for orig, new in zip(original_lines, lines) if orig.rstrip() != orig)

    # Fix tabs
    tab_count = sum(line.count("\t") for line in original_lines)
    lines = fix_indentation(lines)
    changes["tabs"] = tab_count

    # Join for section parsing
    content = "\n".join(lines)

    # Parse and sort sections
    sections = parse_yaml_simple(content)
    sorted_sections = OrderedDict()

    for section_name, section_lines in sections.items():
        sorted_lines = sort_section_keys(section_lines)
        if sorted_lines != section_lines:
            changes["sorted"].append(section_name)
        sorted_sections[section_name] = sorted_lines

    # Reconstruct file
    result_lines = []
    for i, (section_name, section_lines) in enumerate(sorted_sections.items()):
        if i > 0:
            result_lines.append("")  # Blank line between sections
        result_lines.append(f"{section_name}:")
        result_lines.extend(section_lines)

    # Ensure file ends with newline
    if result_lines and result_lines[-1] != "":
        result_lines.append("")

    # Check if modified
    new_content = "\n".join(result_lines)
    changes["modified"] = new_content != content

    if changes["modified"]:
        total_changes = changes["trailing_ws"] + changes["tabs"] + len(changes["sorted"])
        print(f"  {filepath}: {total_changes} changes")

        if changes["trailing_ws"]:
            print(f"    - Removed {changes['trailing_ws']} trailing whitespace")
        if changes["tabs"]:
            print(f"    - Converted {changes['tabs']} tabs to spaces")
        if changes["sorted"]:
            print(f"    - Sorted keys in sections: {', '.join(changes['sorted'])}")

        if not dry_run:
            try:
                with open(filepath, "w") as f:
                    f.write(new_content)
                print(f"    ✓ Applied changes")
            except Exception as e:
                print(f"    ✗ Error writing {filepath}: {e}")
        else:
            print(f"    (dry-run, not applied)")

    return changes


def main():
    parser = argparse.ArgumentParser(description="Tidy YAML configuration files")
    parser.add_argument("--config-dir", default="config", help="Config directory to process")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    dry_run = not args.apply

    print("=" * 70)
    print("YAML FILE TIDYING")
    print("=" * 70)
    print(f"Config directory: {config_dir.absolute()}")
    print(f"Mode: {'DRY-RUN (preview only)' if dry_run else 'APPLY CHANGES'}")
    print("=" * 70)

    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        return 1

    yaml_files = list(config_dir.rglob("*.yaml")) + list(config_dir.rglob("*.yml"))
    print(f"\nFound {len(yaml_files)} YAML files\n")

    total_stats = {"modified": 0, "trailing_ws": 0, "tabs": 0, "sorted_sections": 0}

    for filepath in yaml_files:
        changes = process_yaml_file(filepath, dry_run)
        if changes["modified"]:
            total_stats["modified"] += 1
            total_stats["trailing_ws"] += changes["trailing_ws"]
            total_stats["tabs"] += changes["tabs"]
            total_stats["sorted_sections"] += len(changes["sorted"])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Files modified: {total_stats['modified']}")
    print(f"  Trailing whitespace removed: {total_stats['trailing_ws']}")
    print(f"  Tabs converted to spaces: {total_stats['tabs']}")
    print(f"  Sections sorted: {total_stats['sorted_sections']}")

    if dry_run:
        print("\nThis was a dry-run. Use --apply to make changes.")

    return 0


from typing import List, Dict, Any

if __name__ == "__main__":
    sys.exit(main())
