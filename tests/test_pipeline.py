"""Tests: pipeline, HPO, SAFuse flagship, registry, config (author: 晨星)."""

import numpy as np
import pytest

from domainforge.core.config import Config
from domainforge.core.errors import UnknownMethodError
from domainforge.data.synthetic import make_covariate_shift
from domainforge.fusion.flagship import SafuseAdapter
from domainforge.hpo.search import hpo_method
from domainforge.pipeline.pipeline import (
    aggregate,
    benchmark,
    format_table,
    run,
    save_rows,
)
from domainforge.registry import build_method

SPLIT = make_covariate_shift(seed=13)


class TestRegistry:
    def test_all_core_methods_buildable(self):
        for name in ["source_only", "coral", "tca", "jda", "kliep"]:
            assert build_method(name, seed=1) is not None

    def test_unknown_raises_e201(self):
        with pytest.raises(UnknownMethodError):
            build_method("nope")

    def test_adapt_coral_requires_optional_flag(self):
        with pytest.raises(UnknownMethodError):
            build_method("adapt_coral")


class TestConfig:
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ENV_DOMAINFORGE_SEED", "777")
        monkeypatch.setenv("ENV_DOMAINFORGE_HPO_TRIALS", "3")
        cfg = Config()
        assert cfg.seed == 777 and cfg.hpo_trials == 3

    def test_bad_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("ENV_DOMAINFORGE_SEED", "not-an-int")
        assert Config().seed == 42


class TestPipeline:
    def test_run_row_ok(self):
        row = run(SPLIT, "coral", seed=1)
        assert row.status == "ok"
        assert 0.0 <= row.accuracy <= 1.0

    def test_run_error_row(self):
        broken = type(SPLIT)(
            "broken",
            SPLIT.X_source,
            SPLIT.y_source,
            SPLIT.X_target[:, :2],
            SPLIT.y_target_private,
            SPLIT.X_target_val,
            SPLIT.y_target_val,
        )
        row = run(broken, "coral", seed=1)
        assert row.status == "error"

    def test_benchmark_deterministic(self):
        a = benchmark([SPLIT], ("core",), seed=1)
        b = benchmark([SPLIT], ("core",), seed=1)
        assert [r.accuracy for r in a] == [r.accuracy for r in b]

    def test_aggregate_keys(self):
        rows = benchmark([SPLIT], ("core",), seed=1)
        agg = aggregate(rows)
        assert set(agg) == {"source_only", "coral", "tca", "jda", "kliep"}

    def test_format_table_alignment(self):
        rows = benchmark([SPLIT], ("source_only",), seed=1)
        table = format_table(rows)
        assert table.splitlines()[0].startswith("dataset")

    def test_save_rows_json(self, tmp_path):
        rows = benchmark([SPLIT], ("source_only",), seed=1)
        p = save_rows(rows, tmp_path / "b.json")
        assert p.exists() and "aggregate" in p.read_text(encoding="utf-8")


class TestHpo:
    def test_hpo_coral_returns_params(self):
        params, val = hpo_method("coral", SPLIT, n_trials=3, seed=1)
        assert "clf_C" in params and 0.0 <= val <= 1.0

    def test_hpo_unknown_method_raises(self):
        with pytest.raises(UnknownMethodError):
            hpo_method("nope", SPLIT, n_trials=2)


class TestSafuse:
    def test_fit_predict_and_mode(self):
        m = SafuseAdapter(do_hpo=False, seed=1).fit(SPLIT)
        pred = m.predict(SPLIT.X_target)
        assert pred.shape == (SPLIT.X_target.shape[0],)
        assert m.mode_ in {"ensemble", "fallback_best_single", "fallback_source_only"}

    def test_fallback_is_valid_mode(self):
        # no members at all -> graceful fallback to source-only anchor
        m = SafuseAdapter(members=[], do_hpo=False, seed=1).fit(SPLIT)
        assert m.mode_ == "fallback_source_only"
        pred = m.predict(SPLIT.X_target)
        assert pred.shape[0] == SPLIT.X_target.shape[0]

    def test_proba_rows_sum_to_one(self):
        m = SafuseAdapter(members=["coral", "tca"], do_hpo=False, seed=1).fit(SPLIT)
        P = m.predict_proba(SPLIT.X_target[:20])
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-5)

    def test_member_tracking(self):
        m = SafuseAdapter(members=["coral", "kliep"], do_hpo=False, seed=1).fit(SPLIT)
        assert set(m.member_val_acc_) == {"coral", "kliep"}

    def test_deterministic(self):
        a = SafuseAdapter(do_hpo=False, seed=3).fit(SPLIT).predict(SPLIT.X_target)
        b = SafuseAdapter(do_hpo=False, seed=3).fit(SPLIT).predict(SPLIT.X_target)
        assert np.array_equal(a, b)
