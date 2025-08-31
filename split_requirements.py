#!/usr/bin/env python3
"""
Split requirements.txt into two files: local references (lines containing '@') and others.

Usage:
    python split_requirements.py [--input requirements.txt] [--local requirements.local.txt] [--remote requirements.pypi.txt]
"""

from pathlib import Path
import argparse


def split_requirements(input_path: str, local_out: str, remote_out: str) -> None:
    input_path = Path(input_path)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    local_names = []
    seen = set()
    remote_lines = []

    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            # Preserve blank lines in remote output
            if line.strip() == "":
                remote_lines.append(line)
                continue

            stripped = line.lstrip()
            # Preserve comment lines (start with # or //) in remote output
            if stripped.startswith('#') or stripped.startswith('//'):
                remote_lines.append(line)
                continue

            # Lines containing '@' are treated as local references (file:// or VCS links)
            if '@' in line:
                # Extract the module/name portion left of the '@'
                name = line.split('@', 1)[0].strip()
                # Deduplicate while preserving order
                if name and name not in seen:
                    seen.add(name)
                    local_names.append(name)
            else:
                remote_lines.append(line)

    # Write local output as one module name per line (no comments/blank lines)
    Path(local_out).write_text('\n'.join(local_names) + ("\n" if local_names else ""), encoding='utf-8')
    Path(remote_out).write_text(''.join(remote_lines), encoding='utf-8')

    print(f"Wrote {len(local_names)} module names to {local_out}")
    non_empty = lambda ls: len([L for L in ls if L.strip() != ""])
    print(f"Wrote {non_empty(remote_lines)} non-empty lines to {remote_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Split requirements into local and remote files')
    parser.add_argument('--input', '-i', default='requirements.txt', help='Input requirements file')
    parser.add_argument('--local', '-l', default='requirements.local.txt', help='Output file for local references (containing @)')
    parser.add_argument('--remote', '-r', default='requirements.pypi.txt', help='Output file for remote/pypi references')
    args = parser.parse_args()

    split_requirements(args.input, args.local, args.remote)


if __name__ == '__main__':
    main()
