# METR-LA data preparation

Raw data are not excluded by `.gitignore`. This directory must contain:

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `metr-la.h5` | 11,361,867 | `e00a068ca5ec0abcf225b5a044bf962cc1c32c4fef31bad8187c04f342ef0990` |
| `adj_mx.pkl` | 680,459 | `a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3` |

Follow the [official DCRNN data preparation instructions](https://github.com/liyaguang/DCRNN#data-preparation) to obtain `metr-la.h5`. Use `data/sensor_graph/adj_mx.pkl` from DCRNN for the adjacency information. Place both files here, then follow the parent README.

The `.pkl` file uses Python pickle; load only files from trusted original sources. Data use and redistribution remain subject to the provider's terms. A source-code license does not replace data permissions.

Dataset reference: Yaguang Li, Rose Yu, Cyrus Shahabi, and Yan Liu, *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*, ICLR 2018.
