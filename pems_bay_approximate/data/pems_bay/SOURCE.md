# PEMS-BAY source record

Downloaded from the open Zenodo record:

- DOI: https://doi.org/10.5281/zenodo.4263971
- Record: https://zenodo.org/records/4263971
- Download date: 2026-07-28

Files and verified MD5 checksums:

| File | Size | MD5 |
| --- | ---: | --- |
| `pems-bay.h5` | 135,930,936 bytes | `bfa47e6ee7cc2d665e9b62c0c95c0b41` |
| `adj_mx_bay.pkl` | 1,681,480 bytes | `55d25daceac847312b7748c59ded6f77` |

The HDF5 file contains 52,116 five-minute observations from 2017-01-01
00:00:00 through 2017-06-30 23:55:00 for 325 sensors. The adjacency matrix is
325 by 325 and uses the same ordered sensor IDs. The downloaded speed matrix
has no NaNs and contains 521 zero entries; the experiment treats zero speed as
missing and interpolates it using the same rule as the METR-LA experiment.

The HDF5 frame was written by pandas 0.15.2. `pems_nsmp/models_and_methods.py` includes a
read-only PyTables compatibility path because pandas 3 cannot directly parse
the legacy byte-valued frame metadata.

The machine-readable validation results are saved in `data_profile.json`.

## Preparing a checkout

Raw `.h5` and `.pkl` files are not excluded by `.gitignore`. If missing, obtain
`pems-bay.h5` and `adj_mx_bay.pkl` from the record above and place them beside
this document. The direct record file links are:

- https://zenodo.org/records/4263971/files/pems-bay.h5?download=1
- https://zenodo.org/records/4263971/files/adj_mx_bay.pkl?download=1

Verify the file sizes and MD5 values above. The HDF5 file exceeds GitHub's
100 MiB regular-file limit. Dataset terms are separate from the project's
source-code license; this repository does not assert redistribution rights
for the external dataset.
