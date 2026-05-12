"""Utility helpers: logging configuration and formatted report printer."""

from __future__ import annotations

import logging
import sys
from typing import Dict


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the root package logger.

    Args:
        level: Python logging level (default: INFO).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logger = logging.getLogger("shell_hackathon")
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger with the shared root configuration."""
    return logging.getLogger(name)


def print_report(oof_scores: Dict[str, float], val_scores: Dict[str, float]) -> None:
    """Print a formatted MAPE report table to stdout.

    Args:
        oof_scores: Mapping of target name → OOF MAPE (float).
        val_scores: Mapping of target name → hold-out validation MAPE (float).
    """
    targets = [f"BlendProperty{i}" for i in range(1, 11)]
    col_w = 16

    header = f"{'Target':<{col_w}} {'OOF MAPE':>10} {'Val MAPE':>10}"
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)

    oof_vals = []
    val_vals = []
    for t in targets:
        oof = oof_scores.get(t, float("nan"))
        val = val_scores.get(t, float("nan"))
        oof_vals.append(oof)
        val_vals.append(val)
        print(f"{t:<{col_w}} {oof:>10.4f} {val:>10.4f}")

    print(sep)
    mean_oof = sum(v for v in oof_vals if v == v) / max(len(oof_vals), 1)
    mean_val = sum(v for v in val_vals if v == v) / max(len(val_vals), 1)
    print(f"{'Mean':<{col_w}} {mean_oof:>10.4f} {mean_val:>10.4f}")
    print(sep)
    print()
