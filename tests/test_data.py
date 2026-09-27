"""Tests: data layer contracts and loaders (author: 晨星)."""

import numpy as np
import pytest

from domainforge.core.errors import DimensionMismatchError, InvalidDataError
from domainforge.data.loaders import load_csv, load_npz
from domainforge.data.synthetic import make_all, make_covariate_shift, validate_split


class TestSynthetic:
    def test_all_splits_deterministic(self):
        a = make_all(n_target_val=10, seed=5)
        b = make_all(n_target_val=10, seed=5)
        for sa, sb in zip(a, b, strict=False):
            assert np.array_equal(sa.X_source, sb.X_source)
            assert np.array_equal(sa.y_target_private, sb.y_target_private)

    def test_val_test_disjoint(self):
        s = make_covariate_shift(n_val=15, seed=2)
        # rows must not overlap between the tiny val sample and the test pool
        v = {tuple(np.round(r, 9)) for r in s.X_target_val}
        t = {tuple(np.round(r, 9)) for r in s.X_target}
        assert not (v & t)

    def test_val_has_both_classes(self):
        for s in make_all(n_target_val=20, seed=1):
            assert len(np.unique(s.y_target_val)) >= 2

    def test_validate_split_rejects_dim_mismatch(self):
        s = make_covariate_shift(seed=1)
        bad = type(s)(
            "bad",
            s.X_source[:, :3],
            s.y_source,
            s.X_target,
            s.y_target_private,
            s.X_target_val,
            s.y_target_val,
        )
        with pytest.raises((InvalidDataError, DimensionMismatchError)):
            validate_split(bad)


class TestLoaders:
    def test_npz_roundtrip(self, tmp_path):
        s = make_covariate_shift(seed=4)
        p = tmp_path / "toy.npz"
        np.savez(p, Xs=s.X_source, ys=s.y_source, Xt=s.X_target, yt=s.y_target_private)
        loaded = load_npz(p, n_target_val=10, seed=4)
        assert loaded.X_source.shape == s.X_source.shape
        assert loaded.X_target_val.shape[0] == 10

    def test_npz_missing_raises(self, tmp_path):
        with pytest.raises(InvalidDataError):
            load_npz(tmp_path / "nope.npz")

    def test_csv_roundtrip(self, tmp_path):
        s = make_covariate_shift(seed=6)
        p = tmp_path / "toy.csv"
        with p.open("w", encoding="utf-8") as fh:
            fh.write(
                ",".join([f"feature_{i}" for i in range(s.X_source.shape[1])])
                + ",label,domain\n"
            )
            for x, y in zip(s.X_source, s.y_source, strict=False):
                fh.write(",".join(f"{float(v):.10g}" for v in x) + f",{int(y)},0\n")
            for x, y in zip(s.X_target, s.y_target_private, strict=False):
                fh.write(",".join(f"{float(v):.10g}" for v in x) + f",{int(y)},1\n")
        loaded = load_csv(p, n_target_val=10, seed=6)
        assert loaded.X_source.shape[1] == s.X_source.shape[1]

    def test_csv_bad_header_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("a,b,label\n1,2,0\n", encoding="utf-8")
        with pytest.raises(InvalidDataError):
            load_csv(p)
