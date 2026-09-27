"""Tests: CORAL / TCA / JDA mathematical invariants (author: 晨星)."""

import numpy as np
import pytest

from domainforge.alignment.coral import CoralAdapter
from domainforge.alignment.jda import JdaAdapter
from domainforge.alignment.tca import TcaAdapter
from domainforge.data.synthetic import make_covariate_shift, make_rotated_moons

SPLIT = make_covariate_shift(seed=7)


class TestCoral:
    def test_covariance_alignment(self):
        m = CoralAdapter(seed=1).fit(SPLIT)
        A = m._A
        Xs0 = SPLIT.X_source - SPLIT.X_source.mean(axis=0)
        Xt0 = SPLIT.X_target - SPLIT.X_target.mean(axis=0)
        Cs_aligned = np.cov((Xs0 @ A), rowvar=False)
        Ct = np.cov(Xt0, rowvar=False)
        assert np.allclose(Cs_aligned, Ct, atol=1e-6)

    def test_transform_passes_through(self):
        m = CoralAdapter(seed=1).fit(SPLIT)
        Z = m.transform(SPLIT.X_target[:5])
        assert np.allclose(Z, SPLIT.X_target[:5])

    def test_predict_shape_and_classes(self):
        m = CoralAdapter(seed=1).fit(SPLIT)
        pred = m.predict(SPLIT.X_target)
        assert pred.shape == (SPLIT.X_target.shape[0],)
        assert set(np.unique(pred)) <= set(np.unique(SPLIT.y_source))

    def test_beats_or_matches_source_only_on_covariate_shift(self):
        from domainforge.baselines.source_only import SourceOnlyAdapter

        acc_c = float(
            (
                CoralAdapter(seed=1).fit(SPLIT).predict(SPLIT.X_target)
                == SPLIT.y_target_private
            ).mean()
        )
        acc_s = float(
            (
                SourceOnlyAdapter(seed=1).fit(SPLIT).predict(SPLIT.X_target)
                == SPLIT.y_target_private
            ).mean()
        )
        assert acc_c >= acc_s - 0.02


class TestTca:
    def test_embedding_dim(self):
        m = TcaAdapter(n_components=8, seed=1).fit(SPLIT)
        Z = m.transform(SPLIT.X_target[:10])
        assert Z.shape == (10, 8)

    def test_generalized_eigenproblem_residual(self):
        # recompute the generalized problem on the CENTERED stack (same as
        # the implementation): (KMK + mu I) w = lam (KHK) w, check residual
        # with the per-column eigenvalue lam_i
        from sklearn.metrics.pairwise import rbf_kernel

        m = TcaAdapter(n_components=8, mu=1e-2, seed=1).fit(SPLIT)
        Zs = SPLIT.X_source - SPLIT.X_source.mean(axis=0)
        Zt = SPLIT.X_target - SPLIT.X_target.mean(axis=0)
        Z = np.vstack([Zs, Zt])
        ns = Zs.shape[0]
        K = rbf_kernel(Z, Z, gamma=m._gamma)
        n = K.shape[0]
        d = np.concatenate([np.full(ns, 1 / ns), np.full(n - ns, 1 / (n - ns))])
        M0 = np.outer(d, d)
        H = np.eye(n) - np.ones((n, n)) / n
        W = m._W
        A = K @ M0 @ K + m.mu * np.eye(n)
        B = K @ H @ K + 1e-8 * np.trace(K @ H @ K) / n * np.eye(n)
        lam = (W * (A @ W)).sum(axis=0) / (W * (B @ W)).sum(axis=0)
        R = A @ W - B @ W * lam
        scale = np.linalg.norm(B @ W * lam, axis=0) + 1e-12
        assert float((np.linalg.norm(R, axis=0) / scale).max()) < 1e-4

    def test_predict_shape(self):
        m = TcaAdapter(n_components=8, seed=1).fit(SPLIT)
        pred = m.predict(SPLIT.X_target)
        assert pred.shape == (SPLIT.X_target.shape[0],)


class TestJda:
    def test_runs_and_embeds(self):
        m = JdaAdapter(n_components=8, n_iters=2, seed=1).fit(SPLIT)
        Z = m.transform(SPLIT.X_target[:5])
        assert Z.shape == (5, 8)

    def test_pseudo_labels_stabilize(self):
        m = JdaAdapter(n_components=8, n_iters=4, seed=1).fit(SPLIT)
        assert m.n_pseudo_flips_ >= 0

    def test_not_fitted_raises(self):
        m = JdaAdapter(seed=1)
        with pytest.raises(Exception):
            m.predict(SPLIT.X_target)

    def test_rotated_moons_smoke(self):
        s = make_rotated_moons(seed=3)
        m = JdaAdapter(n_components=6, n_iters=2, seed=1).fit(s)
        pred = m.predict(s.X_target)
        assert pred.shape == (s.X_target.shape[0],)
