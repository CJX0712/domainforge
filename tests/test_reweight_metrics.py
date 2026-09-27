"""Tests: KLIEP reweighting, source_only baseline, metrics (author: 晨星)."""

import numpy as np

from domainforge.alignment.reweight import KliepAdapter
from domainforge.baselines.source_only import SourceOnlyAdapter
from domainforge.data.synthetic import make_covariate_shift
from domainforge.eval.metrics import evaluate, linear_mmd2

SPLIT = make_covariate_shift(seed=11)


class TestKliep:
    def test_weights_positive_bounded_nonconstant(self):
        m = KliepAdapter(seed=1).fit(SPLIT)
        w = m.weights_
        assert (w > 0).all() and w.max() <= 8.0
        assert float(w.std()) > 1e-6  # weights must actually discriminate

    def test_predict_shape(self):
        m = KliepAdapter(seed=1).fit(SPLIT)
        pred = m.predict(SPLIT.X_target)
        assert pred.shape == (SPLIT.X_target.shape[0],)

    def test_transform_standardizes(self):
        m = KliepAdapter(seed=1).fit(SPLIT)
        Z = m.transform(SPLIT.X_target)
        assert abs(float(Z.mean())) < 1.0  # roughly standardized


class TestSourceOnly:
    def test_baseline_runs(self):
        m = SourceOnlyAdapter(seed=1).fit(SPLIT)
        pred = m.predict(SPLIT.X_target)
        acc, f1 = evaluate(SPLIT.y_target_private, pred)
        assert 0.0 <= acc <= 1.0 and 0.0 <= f1 <= 1.0


class TestMetrics:
    def test_mmd_same_domain_small(self):
        X = np.random.default_rng(0).normal(size=(200, 5))
        assert linear_mmd2(X, X) < 1e-8

    def test_mmd_shifted_larger(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(200, 5))
        B = rng.normal(size=(200, 5)) + 3.0
        assert linear_mmd2(A, B) > 1.0

    def test_evaluate_perfect(self):
        acc, f1 = evaluate(np.array([0, 1, 2]), np.array([0, 1, 2]))
        assert acc == 1.0 and f1 == 1.0
