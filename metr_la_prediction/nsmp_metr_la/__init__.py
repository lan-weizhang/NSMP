"""METR-LA comparison of Local, FedAvg, FedRep, and NSMP."""

from .data import FederatedTimeSeries, load_metr_la
from .methods import (
    MethodResult,
    fit_fedavg,
    fit_fedrep,
    fit_local,
    fit_nsmp,
)

__all__ = [
    "FederatedTimeSeries",
    "MethodResult",
    "fit_fedavg",
    "fit_fedrep",
    "fit_local",
    "fit_nsmp",
    "load_metr_la",
]
