"""Stable error taxonomy for DomainForge (author: 晨星).

E100-E199  data / input errors
E200-E299  adaptation method errors
E300-E399  HPO errors
E400-E499  benchmark / aggregation errors
E500-E599  optional backend / environment errors
"""

from __future__ import annotations


class DomainForgeError(Exception):
    code = "E000"
    message = "base error"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code}] {self.message}"


class InvalidDataError(DomainForgeError):
    code = "E100"
    message = "invalid data: domain split malformed or empty"


class DimensionMismatchError(DomainForgeError):
    code = "E101"
    message = "feature dimension mismatch between source and target"


class AdaptationError(DomainForgeError):
    code = "E200"
    message = "adaptation method failed"


class UnknownMethodError(DomainForgeError):
    code = "E201"
    message = "unknown adaptation method"


class HPOError(DomainForgeError):
    code = "E300"
    message = "hyper-parameter search failed"


class BenchmarkError(DomainForgeError):
    code = "E400"
    message = "benchmark aggregation failed"


class BackendUnavailableError(DomainForgeError):
    code = "E500"
    message = "optional backend unavailable"
