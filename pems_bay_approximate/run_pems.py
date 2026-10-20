"""Run PEMS-BAY and save only the final tables, raw metrics, and configuration."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from pems_nsmp.summary import SEEDS, run_directory_name, summarize_results

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--seed', type=int, choices=SEEDS, help='Run one seed instead of all three.')
    parser.add_argument('--output-root', type=Path, default=HERE / 'results/latest')
    parser.add_argument('--summarize-only', action='store_true', help='Rebuild tables from saved raw metrics.')
    parser.add_argument('--dry-run', action='store_true', help='Show commands without training or writing results.')
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    seeds = (args.seed,) if args.seed is not None else SEEDS

    def commands(temporary_root: Path):
        for seed in seeds:
            yield [sys.executable, '-u', '-m', 'pems_nsmp.messages',
                   '--seed', str(seed), '--device', args.device,
                   '--output', str(temporary_root / run_directory_name(seed))]

    if args.dry_run:
        if not args.summarize_only:
            for command in commands(output_root / '_temporary'):
                print(subprocess.list2cmdline(command))
        print(f'Final tables and raw metrics: {output_root}')
        return
    if args.summarize_only:
        summarize_results(output_root)
        return

    missing = [name for name in ('pems-bay.h5', 'adj_mx_bay.pkl')
               if not (HERE / 'data/pems_bay' / name).is_file()]
    if missing:
        parser.error('Missing external data: ' + ', '.join(missing)
                     + '. See data/pems_bay/SOURCE.md for download instructions.')

    # Intermediate models, plots and message diagnostics are discarded after aggregation.
    temporary_parent = output_root.parent
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.pems_run_', dir=temporary_parent) as name:
        temporary_root = Path(name).resolve()
        if not temporary_root.is_relative_to(temporary_parent):
            raise ValueError('Temporary outputs must stay inside the results directory.')
        for command in commands(temporary_root):
            print(subprocess.list2cmdline(command), flush=True)
            subprocess.run(command, cwd=HERE, check=True)
        summarize_results(temporary_root, output_root, seeds)


if __name__ == '__main__':
    main()
