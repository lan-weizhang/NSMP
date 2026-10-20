# METR-LA traffic speed prediction

This directory contains the METR-LA experiment for **Nerve-Skeleton Message Passing for Federated Optimization with Overlapping Parameters** (NeurIPS 2026). See the [repository README](../README.md) for the paper and full code structure.

The experiment treats 206 traffic sensors as 206 clients and compares Local, FedAvg, FedRep, and NSMP in prediction accuracy and communication cost.

## 1. Experimental setup

Each client predicts the next speed value from the previous 12 time steps, using 12 lag coefficients and one bias. Data are split chronologically into 70% training, 10% validation, and 20% test.

| Scenario | Retained training data per client |
| --- | --- |
| `low_data_1pct` | 1% of the training split |
| `low_data_0_5pct` | 0.5% of the training split |

Both scenarios retain the full validation and test sets. Zero values are treated as missing. Normalization statistics are computed only from retained training data. All methods use the same splits and training samples.

Configuration: [configs/paper.json](configs/paper.json). Random seed: 2026.

## 2. Environment and execution

Use Python 3.12 with a Conda environment named `env_NSMP`. Dependencies are listed in [requirements.txt](requirements.txt). From this directory:

```powershell
conda create -n env_NSMP python=3.12 -y
conda activate env_NSMP
python -m pip install -r requirements.txt
python run_scenarios.py
```

Run a single scenario:

```powershell
python run_scenarios.py --scenarios low_data_1pct
python run_scenarios.py --scenarios low_data_0_5pct
```

For a quick execution check with fewer clients:

```powershell
python run_scenarios.py --num-clients 20 --scenarios low_data_1pct --output-root results/quick_check
```

This check uses 20 clients and does not reproduce the paper's 206-client experiment. From the repository root, run the paper configuration with:

```powershell
python main_run.py metr
```

## 3. Implementations

All four methods are in [methods.py](nsmp_metr_la/methods.py):

| Method | Function | Parameter sharing |
| --- | --- | --- |
| Local | `fit_local` | Each client fits independently |
| FedAvg | `fit_fedavg` | All model parameters are shared |
| FedRep | `fit_fedrep` | Selected lag coefficients are shared; other coefficients and biases remain private |
| NSMP | `fit_nsmp` | Quadratic messages solve a graph-regularized personalized objective on a similarity spanning tree |

FedAvg and FedRep share one implementation file rather than separate directories.

| File | Purpose |
| --- | --- |
| [run_scenarios.py](run_scenarios.py) | Reads the configuration and runs the scenarios |
| [experiment.py](nsmp_metr_la/experiment.py) | Runs methods, selects validation hyperparameters, evaluates, and saves results |
| [data.py](nsmp_metr_la/data.py) | Loads data, constructs prediction windows, and splits by time |
| [scenarios.py](nsmp_metr_la/scenarios.py) | Retains training samples and recomputes normalization statistics |
| [graph.py](nsmp_metr_la/graph.py) | Constructs the client similarity spanning tree |
| [quadratic.py](nsmp_metr_la/quadratic.py) | Constructs local quadratic objectives |

FedAvg and FedRep use at most 300 rounds and `local_epochs=1`. FedRep selects the number of shared lags from `3/6/9/12` by validation RMSE. NSMP also selects its coupling coefficient on validation data. See the configuration for all candidate values and remaining settings.

## 4. Data

```text
data/
├── metr-la.h5       Traffic speed time series
└── adj_mx.pkl       Sensor adjacency information
```

Raw files are not ignored by Git. Existing files can be used directly; obtain missing files using [data/README.md](data/README.md).

## 5. Outputs

Results are saved to `results/latest/`; repeated runs update the selected scenarios. Saved reference results are in `results/reference/`.

```text
results/latest/
├── all_scenarios_summary.csv
├── scenario_comparison.png
├── low_data_1pct/
└── low_data_0_5pct/
```

| Per-scenario file | Contents |
| --- | --- |
| `summary.csv` | Validation / test metrics, runtime, and communication for all four methods |
| `per_client_metrics.csv` | Client-level prediction metrics |
| `client_training_samples.csv` | Retained training sample counts |
| `method_metadata.json` | Method settings, validation selections, and numerical checks |
| `resolved_config.json` | Run configuration and selected parameters |
| `mst_edges.csv` | Spanning tree edges used by NSMP |
| `histories.json` | Training histories |
| `parameters_*.npy` | Learned model parameters |
| `test_metrics.png` | Test metric comparison |
| `validation_convergence.png` | Validation convergence curves |

Communication counts Float64 numerical payloads, excluding protocol headers and hyperparameter search. NSMP's `communication_rounds=2` denotes the upward and downward phases; synchronous time slots also depend on tree depth.

## 6. Verified results

| Method | Test RMSE: 1% | Test RMSE: 0.5% | Communication (bytes) |
| --- | ---: | ---: | ---: |
| Local | 4.2548 | 4.5508 | 0 |
| FedAvg | 4.1512 | 4.1627 | 12,854,400 |
| FedRep | 4.1595 | 4.1805 | 11,865,600 |
| NSMP | **4.0742** | **4.1106** | **193,520** |

Both scenarios have been run and their test MAE, RMSE, MAPE, and communication costs checked against the saved reference results. Communication costs are identical across the two scenarios.

## 7. Tests

From this directory:

```powershell
python -m pytest -q -p no:cacheprovider tests
```
