# Distributed state estimation

This directory contains the state estimation experiments for **Nerve-Skeleton Message Passing for Federated Optimization with Overlapping Parameters** (NeurIPS 2026). See the [repository README](../README.md) for the paper and full code structure.

EPFL and SoCal real phasor measurements are used to compare NSMP, ADMM, and Gossip on the same weighted least-squares (WLS) objective. A centralized solution provides the accuracy reference.

## 1. Directory structure

```text
state_estimation/
├── README.md
├── run_epfl.py
├── run_socal.py
├── epfl_pmu_state_estimation/
│   ├── prepare_epfl_archive.py  Configuration generation for recovered data
│   ├── epfl_wls/               Shared solvers
│   ├── configs/                Network, column mappings, and Ybus
│   ├── data/
│   ├── tests/
│   └── results/
└── socal28_state_estimation/
    ├── socal_wls/               SoCal adapter and local objectives
    ├── official_repo/           Upstream data, topology, and utilities
    ├── tests/
    └── results/
```

SoCal imports `epfl_wls` from the sibling EPFL directory. Preserve their relative locations.

## 2. Environment and execution

Use Python 3.12 with a Conda environment named `env_NSMP`. The verified runs used NumPy 2.5.1 and pandas 3.0.5. Each dataset directory provides a requirements file. From `state_estimation/`:

```powershell
conda create -n env_NSMP python=3.12 -y
conda activate env_NSMP
python -m pip install -r epfl_pmu_state_estimation/requirements.txt -r socal28_state_estimation/requirements.txt
python run_epfl.py
python run_socal.py
```

Inspect available arguments:

```powershell
python run_epfl.py --help
python run_socal.py --help
```

Both runners are next to this README and resolve data, configuration, and output paths relative to their own location. Shapely, PyArrow, and pytz are included in the SoCal requirements. There is no need to change into a dataset subdirectory.

## 3. Implementations

Both datasets use [epfl_wls/solvers.py](epfl_pmu_state_estimation/epfl_wls/solvers.py):

| Method | Function | Purpose |
| --- | --- | --- |
| Centralized | `solve_centralized` | Centralized accuracy reference |
| NSMP | `solve_nsmp` | Tree-based quadratic message elimination and recovery |
| ADMM | `solve_consensus_admm` | Consensus ADMM over overlapping coordinates |
| Gossip | `solve_decentralized_gossip` | Decentralized gradient tracking |

| File | Purpose |
| --- | --- |
| [run_epfl.py](run_epfl.py) | Loads EPFL configuration, constructs objectives, runs methods, and saves results |
| [data.py](epfl_pmu_state_estimation/epfl_wls/data.py) | Parses and synchronizes EPFL measurements |
| [model.py](epfl_pmu_state_estimation/epfl_wls/model.py) | Constructs measurement models and local WLS factors |
| [quadratic.py](epfl_pmu_state_estimation/epfl_wls/quadratic.py) | Represents and manipulates quadratic objectives |
| [run_socal.py](run_socal.py) | Runs SoCal and aggregates snapshot results |
| [adapter.py](socal28_state_estimation/socal_wls/adapter.py) | Loads SoCal measurements and topology and constructs local objectives |

ADMM and Gossip are the comparison methods here. FedAvg and FedRep are used in the METR-LA and PEMS-BAY prediction experiments.

## 4. Data and settings

| Setting | EPFL | SoCal |
| --- | --- | --- |
| Data directory | `epfl_pmu_state_estimation/data/epfl_wayback_2014_08_04_19h/` | `socal28_state_estimation/official_repo/sample_dataset/` |
| Configuration | EPFL `configs/epfl_wayback_2014_08_04_19h/` | Topology and phasor mappings in `official_repo/` |
| Snapshots | 4, `stride=10000` | 3 |
| ADMM iterations | 1000 | 2000 |
| Gossip iterations | 3000 | 5000 |
| Gossip learning-rate factor | 0.4 | 0.05 |

ADMM and Gossip use fixed iteration budgets. NSMP rounds follow the upward / downward tree schedule. These are the default runner settings and are saved in `run_metadata.json`.

EPFL uses a recovered synchronized segment from five PMUs; provenance and checksums are documented in its data directory. The SoCal dataset is called SoCal 28-Bus, but this experiment solves the A-side subproblem, which has 12 electrical nodes and 72 real state coordinates after merging and pruning.

### Obtaining missing files

Raw data and upstream SoCal code are not excluded by `.gitignore`. Existing files can be used directly.

- EPFL: follow the [data documentation](epfl_pmu_state_estimation/data/epfl_wayback_2014_08_04_19h/README.md) to obtain five `.asc` files and `CFG.dat`, place them beside that document, and verify SHA-256 hashes. Network and column mappings are provided in `configs/`.
- SoCal: if `official_repo/` is missing, run the following from `state_estimation/`:

```powershell
git clone https://github.com/caltech-netlab/digital-twin-dataset.git socal28_state_estimation/official_repo
git -C socal28_state_estimation/official_repo checkout 607eb77156012038746ab166d76190adae56a6fb
```

This commit matches the verified upstream source. Obtain phasor files from the [public SoCal sample](https://caltech.box.com/s/5baxy2ogbalqohpidh1lyxgnnxmv5tuc) and place them in `socal28_state_estimation/official_repo/sample_dataset/phasors/`, using the filenames and sizes in [box_phasor_manifest.json](socal28_state_estimation/box_phasor_manifest.json), including `t.csv`.

No standalone license file was found in the upstream SoCal repository. Its code and data remain outside the scope of any project source-code license; third-party use and distribution terms are determined by the provider.

## 5. Outputs and metrics

Outputs are saved to:

```text
epfl_pmu_state_estimation/results/latest/
socal28_state_estimation/results/latest/
```

Repeated runs update the corresponding directory. Each dataset saves:

| File | Contents |
| --- | --- |
| `summary.csv` | Full summary including Centralized and all methods: accuracy, runtime, rounds, and communication |
| `summary_compact.csv` | Paper-format table for NSMP, ADMM, and Gossip: Dataset, Method, Error metric, Hop rounds/iterations, Bytes |
| `per_snapshot_results.csv` | Detailed metrics for each snapshot and method |
| `run_metadata.json` | Data range, iteration budgets, and configuration |

The compact table uses current results, with errors rounded to three significant digits. EPFL reports objective gap; SoCal reports state RMSE. Bytes equal the number of Float64 scalars multiplied by eight.

`communicated_scalars` counts real scalars. `rounds` means iterations for iterative methods and synchronous communication rounds for NSMP.

EPFL uses a shifted quadratic objective gap relative to the centralized optimum to avoid subtracting large numbers. SoCal uses state RMSE relative to the centralized solution in per unit. These metrics are not directly comparable.

## 6. Verified results

| Dataset | NSMP accuracy | Synchronous rounds | Communication (bytes) |
| --- | ---: | ---: | ---: |
| EPFL | Objective gap `5.9399e-14` | 4 | 3,296 |
| SoCal | State RMSE `8.0913e-9` | 8 | 9,064 |

These are the latest complete runs in the verified environment. Both datasets retain only `results/latest/` and update it on subsequent runs.

## 7. Tests

Run the following separately from `epfl_pmu_state_estimation/` and `socal28_state_estimation/`:

```powershell
python -m pytest -q -p no:cacheprovider tests
```
