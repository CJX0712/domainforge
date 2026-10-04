"""Optuna-backed per-method HPO on the tiny labeled target val set (author: 晨星).

Objective = val accuracy of the method with candidate params. Deterministic:
TPESampler(seed=config.seed). ``timeout<=0`` means "no timeout" (Optuna 5.x
rejects timeout=0 - SOP pitfall).
"""

from __future__ import annotations

import optuna

from domainforge.core.config import CONFIG
from domainforge.core.errors import HPOError, UnknownMethodError
from domainforge.core.types import DomainSplit
from domainforge.registry import build_method

optuna.logging.set_verbosity(optuna.logging.WARNING)

# search spaces: (low, high) or explicit choices per parameter
SPACES: dict[str, dict[str, tuple]] = {
    "source_only": {"clf_C": (1e-2, 100.0)},
    "coral": {"clf_C": (1e-2, 100.0)},
    "tca": {"n_components": (4, 32, "int"), "mu": (1e-4, 1.0), "clf_C": (1e-2, 100.0)},
    "jda": {
        "n_components": (4, 32, "int"),
        "mu": (1e-4, 1.0),
        "n_iters": (1, 4, "int"),
        "clf_C": (1e-2, 100.0),
    },
    "kliep": {"disc_C": (1e-2, 100.0), "clip_hi": (2.0, 20.0), "clf_C": (1e-2, 100.0)},
}


def _suggest(trial: optuna.Trial, name: str, spec: tuple) -> float | int:
    if len(spec) == 3 and spec[2] == "int":
        return trial.suggest_int(name, int(spec[0]), int(spec[1]))
    lo, hi = spec[0], spec[1]
    return float(trial.suggest_float(name, lo, hi, log=(lo > 0 and hi / lo > 100)))


def hpo_method(
    method_name: str,
    split: DomainSplit,
    n_trials: int | None = None,
    seed: int | None = None,
) -> tuple[dict, float]:
    """Run TPE over the method's space; return (best_params, best_val_acc).

    Val accuracy is measured on the tiny labeled target validation sample -
    the only target supervision any method is allowed to see.
    """
    n_trials = n_trials if n_trials is not None else CONFIG.hpo_trials
    seed = CONFIG.seed if seed is None else seed
    if method_name not in SPACES:
        raise UnknownMethodError(f"no HPO space for {method_name}")
    space = SPACES[method_name]
    y_val = split.y_target_val

    def objective(trial: optuna.Trial) -> float:
        params = {k: _suggest(trial, k, spec) for k, spec in space.items()}
        try:
            m = build_method(method_name, params=params, seed=seed)
            m.fit(split)
            pred = m.predict(split.X_target_val)
            return float((pred == y_val).mean())
        except Exception:
            return 0.0  # failed configs lose, never crash the study

    try:
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        kw: dict = {"n_trials": max(1, n_trials)}
        if CONFIG.hpo_timeout and CONFIG.hpo_timeout > 0:
            kw["timeout"] = CONFIG.hpo_timeout
        study.optimize(objective, **kw)
    except optuna.exceptions.ExperimentalWarning:  # pragma: no cover
        raise HPOError("optuna failed") from None
    except Exception as exc:  # pragma: no cover - defensive
        raise HPOError(f"optuna failed: {exc}") from exc
    return dict(study.best_params), float(study.best_value)
