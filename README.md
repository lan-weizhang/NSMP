# NSMP: Nerve-Skeleton Message Passing for Federated Optimization with Overlapping Parameters

Official implementation of **Nerve-Skeleton Message Passing for Federated Optimization with Overlapping Parameters** (NeurIPS 2026).

## 1. Environment

MATLAB R2023a and Python 3.12. Python dependencies:

```text
numpy>=2.0
scipy>=1.15
pandas>=3.0
tables>=3.10
matplotlib>=3.9
torch>=2.3
networkx
shapely
pyarrow
pytz
python-dateutil
pytest>=8.0
```

Install from the repository root:

```powershell
python -m pip install -r requirements.txt
```

The PEMS-BAY data file is managed with Git LFS. After cloning:

```powershell
git lfs install
git lfs pull
```

## 2. Running the experiments

Run from the repository root. Without arguments, `main_run.py` runs only the six-client MATLAB experiment. MATLAB must be available as `matlab`, or specified with `--matlab`.

| Experiment | Command |
| --- | --- |
| Six-client synthetic | `python main_run.py` |
| METR-LA | `python main_run.py metr` |
| EPFL | `python main_run.py epfl` |
| SoCal | `python main_run.py socal` |
| PEMS-BAY | `python main_run.py pems` |

## 3. Experiments

### 3.1 Six-client MATLAB experiment

This experiment evaluates the communication cost of NSMP, ADMM, and Gossip on a synthetic optimization problem with six clients and overlapping parameters. In MATLAB, change to `synthetic_six_client/` and run:

```matlab
run_all_experiments
```

| Method | Rounds | Messages | Scalars | Bytes (F64) |
| --- | --- | --- | --- | --- |
| **NSMP** | **4** | **10** | **20** | **160** |
| Gossip | 23 | 230 | 230 | 1,840 |
| ADMM | 30 | 300 | 300 | 2,400 |

### 3.2 METR-LA traffic speed prediction

This experiment evaluates traffic speed prediction accuracy and communication cost with limited local training data on METR-LA. From `metr_la_prediction/`:

```powershell
python run_scenarios.py
```

| Method | Test RMSE: 1% | Test RMSE: 0.5% | Bytes (F64) |
| --- | --- | --- | --- |
| Local | 4.2548 | 4.5508 | 0 |
| FedAvg | 4.1512 | 4.1627 | 12,854,400 |
| FedRep | 4.1595 | 4.1805 | 11,865,600 |
| **NSMP** | **4.0742** | **4.1106** | **193,520** |

### 3.3 EPFL / SoCal state estimation

This experiment evaluates distributed weighted least-squares state estimation accuracy and communication cost using real EPFL and SoCal phasor measurements. From `state_estimation/`:

```powershell
python run_epfl.py
python run_socal.py
```

| Dataset | Method | Metric | Error | Hop rounds / iterations | Bytes (F64) |
| --- | --- | --- | --- | --- | --- |
| **EPFL** | **NSMP** | **Objective gap** | **5.94e-14** | **4** | **3,296** |
| EPFL | ADMM | Objective gap | 1.03e-04 | 1000 | 1,248,192 |
| EPFL | Gossip | Objective gap | 8.03e-01 | 3000 | 4,608,192 |
| **SoCal** | **NSMP** | **State RMSE** | **8.09e-09** | **8** | **9,064** |
| SoCal | ADMM | State RMSE | 4.65e-04 | 2000 | 6,528,000 |
| SoCal | Gossip | State RMSE | 7.09e-01 | 5000 | 21,120,000 |

### 3.4 PEMS-BAY prediction with approximate messages

This experiment evaluates approximate-message NSMP for nonconvex neural network prediction on PEMS-BAY. From `pems_bay_approximate/`:

```powershell
python run_pems.py
```

| Method | Test RMSE (mean ± SD) | Mean bytes (F32) |
| --- | --- | --- |
| Local | 5.941 ± 0.875 | 0 |
| FedAvg | 5.781 ± 0.937 | 490,560 |
| FedRep | 5.681 ± 0.961 | 456,960 |
| **AdaptiveMsg-NSMP** | **5.406 ± 0.703** | **9,568** |

Python results are saved to each experiment's `results/latest/`; MATLAB results are saved under `synthetic_six_client/`. Repeated runs update the corresponding outputs. For data preparation, settings, and output details, see the experiment READMEs: [METR-LA](metr_la_prediction/README.md), [state estimation](state_estimation/README.md), and [PEMS-BAY](pems_bay_approximate/README.md).

## 4. Citation

If you use this code in your research, please cite our paper, **Nerve-Skeleton Message Passing for Federated Optimization with Overlapping Parameters** (NeurIPS 2026).
