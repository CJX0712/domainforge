"""Global configuration with ENV_DOMAINFORGE_* overrides (author: 晨星)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(f"ENV_DOMAINFORGE_{name}", "")
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(f"ENV_DOMAINFORGE_{name}", "")
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass
class Config:
    seed: int = field(default_factory=lambda: _int_env("SEED", 42))
    hpo_trials: int = field(default_factory=lambda: _int_env("HPO_TRIALS", 24))
    hpo_timeout: float = field(default_factory=lambda: _float_env("HPO_TIMEOUT", 0.0))
    n_components: int = field(default_factory=lambda: _int_env("N_COMPONENTS", 16))
    n_target_val: int = field(default_factory=lambda: _int_env("N_TARGET_VAL", 60))
    fuse_noninferior_tol: float = field(
        default_factory=lambda: _float_env("FUSE_NONINFERIOR_TOL", 0.01)
    )


CONFIG = Config()
