"""Method registry: name -> constructor (author: 晨星).

Single place where benchmark/CLI/HPO resolve method names. Unknown names raise
UnknownMethodError (E201). Optional backends are constructed lazily and are
marked ``optional=True`` so the pipeline can degrade to a skipped row.
"""

from __future__ import annotations

from typing import Any

from domainforge.core.errors import UnknownMethodError


def build_method(
    name: str, params: dict[str, Any] | None = None, seed: int = 42, optional: bool = False
):
    params = dict(params or {})
    if name == "source_only":
        from domainforge.baselines.source_only import SourceOnlyAdapter

        return SourceOnlyAdapter(clf_C=params.get("clf_C", 1.0), seed=seed)
    if name == "coral":
        from domainforge.alignment.coral import CoralAdapter

        return CoralAdapter(clf_C=params.get("clf_C", 1.0), seed=seed)
    if name == "tca":
        from domainforge.alignment.tca import TcaAdapter

        return TcaAdapter(
            n_components=int(params.get("n_components", 16)),
            mu=float(params.get("mu", 1e-2)),
            clf_C=params.get("clf_C", 1.0),
            seed=seed,
        )
    if name == "jda":
        from domainforge.alignment.jda import JdaAdapter

        return JdaAdapter(
            n_components=int(params.get("n_components", 16)),
            mu=float(params.get("mu", 1e-2)),
            n_iters=int(params.get("n_iters", 3)),
            clf_C=params.get("clf_C", 1.0),
            seed=seed,
        )
    if name == "kliep":
        from domainforge.alignment.reweight import KliepAdapter

        return KliepAdapter(
            disc_C=params.get("disc_C", 1.0),
            clip_hi=params.get("clip_hi", 8.0),
            clf_C=params.get("clf_C", 1.0),
            seed=seed,
        )
    if name == "adapt_coral":
        if not optional:
            raise UnknownMethodError("adapt_coral must be built with optional=True")
        from domainforge.baselines.adapt_backend import AdaptCoralBackend

        return AdaptCoralBackend(clf_C=params.get("clf_C", 1.0), seed=seed)
    raise UnknownMethodError(f"unknown method: {name}")


CORE_METHODS = ["source_only", "coral", "tca", "jda", "kliep"]
OPTIONAL_METHODS = ["adapt_coral"]
