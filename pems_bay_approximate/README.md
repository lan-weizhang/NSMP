# PEMS-BAY prediction with approximate messages

This directory contains the PEMS-BAY experiment for **Nerve-Skeleton Message Passing for Federated Optimization with Overlapping Parameters** (NeurIPS 2026). See the [repository README](../README.md) for the paper and full code structure.

The experiment compares Local, FedAvg, FedRep, and AdaptiveMsg-NSMP on traffic speed prediction. NSMP uses nonconvex local neural networks, constructing approximate messages through numerical optimization and function fitting.

```text
pems_bay_approximate/
├── run_pems.py                 Single top-level Python entry point
├── README.md
├── requirements.txt
├── pems_nsmp/
│   ├── __init__.py
│   ├── models_and_methods.py   Networks, data processing, Local / FedAvg / FedRep
│   ├── messages.py             Approximate-message NSMP
│   └── summary.py              Result aggregation
├── data/
├── results/
└── tests/
```

## 1. Experimental setup

| Parameter | Value |
| --- | --- |
| Sensors / clients | 30 sensors, six clients, five sensors per client |
| Input / output | Previous 12 speed time steps / next speed time step |
| Training / validation / test windows | 128 / 240 / 480 per client |
| Hidden dimension | 4 |
| FedAvg / FedRep rounds | 140 |
| Shared interface dimension / radius | 2 / 1.0 |
| Interface regularization | 0.01 |
| Random seeds | 20260728, 20260729, 20260730 |

All methods use the same data splits and evaluation data. Candidate message representations are constant, linear, quadratic, and MLP functions, selected using message validation data.

## 2. Environment and execution

Use Python 3.12 and PyTorch 2.3.1 with a Conda environment named `env_NSMP`. Verified runs used CUDA. See [requirements.txt](requirements.txt). From this directory:

```powershell
conda create -n env_NSMP python=3.12 -y
conda activate env_NSMP
python -m pip install -r requirements.txt
python run_pems.py
```

The runner executes all three seeds sequentially and aggregates the results. For CPU execution:

```powershell
python run_pems.py --device cpu
```

Preview commands without training:

```powershell
python run_pems.py --dry-run
```

Run only seed 20260728:

```powershell
python run_pems.py --seed 20260728
```

The runner uses the local PEMS-BAY data and settings above, saving final results to `results/latest/`. Use `--seed` to select one of the listed seeds and `--output-root` to change the output directory. Run `python run_pems.py --help` for all options.

## 3. Implementations

| Component | File | Function / class |
| --- | --- | --- |
| Local | [models_and_methods.py](pems_nsmp/models_and_methods.py) | `run_local` |
| FedAvg | Same file | `run_fedavg` |
| FedRep | Same file | `run_fedrep` |
| Spatiotemporal prediction network | Same file | `GraphTemporalNet` |
| Data loading and client construction | Same file | `load_metr_la`, `prepare_clients` |
| Approximate-message NSMP | [messages.py](pems_nsmp/messages.py) | `run_approx_mlp_nsmp` |
| Approximate local minimization | Same file | `approximate_minimize` |
| Message representation selection | Same file | `fit_adaptive_message` |
| Private model recovery with fixed interfaces | Same file | `refit_private_models_with_fixed_interfaces` |
| Result aggregation | [summary.py](pems_nsmp/summary.py) | Computes means and sample standard deviations from seed results |

FedAvg and FedRep share `models_and_methods.py`; use `run_pems.py` to run the experiment. The generic data loader retains the name `load_metr_la`, but receives PEMS-BAY data in this experiment.

Local trains all parameters independently. FedAvg shares the full network. FedRep shares the temporal encoder and graph layer while keeping private output layers. AdaptiveMsg-NSMP retains private prediction networks and shares low-dimensional interface variables along tree edges.

## 4. Approximate-message procedure

1. Pretrain private prediction networks locally.
2. At each non-root client, sample the bounded interface domain and approximately minimize over private variables and child interfaces for each fixed parent interface.
3. Fit messages to interface / objective samples and select the representation on validation data.
4. Minimize the assembled approximate objective at the root.
5. Pass interfaces downward and recover client variables. With interfaces fixed, reoptimize private models and select checkpoints using local validation data.

Each non-root client uses 48 fitting samples, 16 validation samples, and 16 independent evaluation samples. Per-seed configurations are collected in the final `run_metadata.json`.

## 5. Data and outputs

Ensure the following files are present; raw files are not excluded by `.gitignore`:

```text
data/pems_bay/
├── pems-bay.h5
├── adj_mx_bay.pkl
└── SOURCE.md
```

See [SOURCE.md](data/pems_bay/SOURCE.md) for provenance and checksums.

Only final outputs are retained:

```text
results/
└── latest/
    ├── summary.csv
    ├── summary_compact.csv
    ├── per_run_results.csv
    └── run_metadata.json
```

| File | Contents |
| --- | --- |
| `summary.csv` | Full summary: run counts and means / sample standard deviations of RMSE, MAE, MAPE, communication, and runtime |
| `summary_compact.csv` | Paper-format table: Method, Test RMSE (mean ± sample standard deviation), Bytes |
| `per_run_results.csv` | Raw metrics for each method and seed, supporting verification and reaggregation |
| `run_metadata.json` | Model and message settings, seeds, and device information |

Models, message fitting records, and diagnostic figures are written to a temporary directory and removed after aggregation. Separate seed folders and seed-comparison figures are not retained. Subsequent runs update the four files in `latest/`.

Regenerate summaries from saved metrics without retraining:

```powershell
python run_pems.py --summarize-only
```

## 6. Verified results

| Method | Test RMSE: mean ± sample standard deviation | Mean communication (bytes) |
| --- | ---: | ---: |
| Local | 5.9409 ± 0.8754 | 0 |
| FedAvg | 5.7812 ± 0.9370 | 490,560 |
| FedRep | 5.6811 ± 0.9608 | 456,960 |
| AdaptiveMsg-NSMP | **5.4062 ± 0.7026** | **9,568** |

RMSE, MAE, MAPE, and communication costs for all three seeds were checked against the saved results. All per-run metrics remain available for verification. Communication counts Float32 numerical payloads at four bytes per value, excluding protocol headers. NSMP communication is the sum of one upward transmission of message parameters and one downward transmission of interface values.

## 7. Tests

From this directory:

```powershell
python -m pytest -q -p no:cacheprovider tests
```
