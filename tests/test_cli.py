"""Tests: CLI smoke (author: 晨星)."""

import pytest

from domainforge.cli import main


class TestCli:
    def test_benchmark_subcommand(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ENV_DOMAINFORGE_HPO_TRIALS", "2")
        out = tmp_path / "bench.json"
        rc = main(
            ["benchmark", "--out", str(out), "--seed", "5", "--methods", "source_only", "coral"]
        )
        assert rc == 0
        assert out.exists()

    def test_run_npz(self, tmp_path):
        import numpy as np

        from domainforge.data.synthetic import make_covariate_shift

        s = make_covariate_shift(seed=8)
        p = tmp_path / "toy.npz"
        np.savez(p, Xs=s.X_source, ys=s.y_source, Xt=s.X_target, yt=s.y_target_private)
        rc = main(["run", "--npz", str(p), "--method", "coral"])
        assert rc == 0

    @pytest.mark.slow
    def test_demo_smoke(self, monkeypatch):
        monkeypatch.setenv("ENV_DOMAINFORGE_HPO_TRIALS", "2")
        rc = main(["demo"])
        assert rc == 0

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as e:
            main(["--version"])
        assert e.value.code == 0
        assert "domainforge" in capsys.readouterr().out
