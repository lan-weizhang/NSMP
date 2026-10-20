"""Run synthetic six-client experiments by default, or select a Python experiment."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def commands(experiment: str):
    if experiment == 'synthetic':
        yield HERE / 'synthetic_six_client', [
            '-batch', 'run_all_experiments; close all force; exit;']
    elif experiment == 'metr':
        yield HERE / 'metr_la_prediction', ['run_scenarios.py', '--config', 'configs/paper.json']
    elif experiment == 'epfl':
        yield HERE / 'state_estimation', ['run_epfl.py']
    elif experiment == 'socal':
        yield HERE / 'state_estimation', ['run_socal.py']
    elif experiment == 'pems':
        yield HERE / 'pems_bay_approximate', ['run_pems.py', '--device', 'auto']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', nargs='?', default='synthetic',
                        choices=['synthetic', 'metr', 'epfl', 'socal', 'pems'])
    parser.add_argument('--dry-run', action='store_true', help='Print commands without running experiments.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], help='Override the saved PEMS device (cuda).')
    parser.add_argument('--matlab', help='MATLAB executable name or path; defaults to matlab on PATH.')
    options = parser.parse_args()
    for directory, args in commands(options.experiment):
        if options.device and options.experiment == 'pems':
            args[args.index('--device') + 1] = options.device
        if options.experiment == 'synthetic':
            executable = shutil.which(options.matlab or 'matlab')
            if executable is None and not options.dry_run:
                parser.error('MATLAB was not found. Add it to PATH or specify --matlab with its executable path.')
            command = [executable or options.matlab or 'matlab'] + args
        else:
            command = [sys.executable] + args
        print(f'Working directory: {directory}', flush=True)
        print(subprocess.list2cmdline(command), flush=True)
        if not options.dry_run:
            environment = os.environ.copy()
            local_dependencies = HERE / '.runtime' / 'socal_dependencies'
            if options.experiment == 'socal' and local_dependencies.is_dir():
                environment['PYTHONPATH'] = os.pathsep.join(filter(None, [
                    str(local_dependencies), environment.get('PYTHONPATH', '')]))
            subprocess.run(command, cwd=directory, check=True, env=environment)



if __name__ == '__main__':
    main()
