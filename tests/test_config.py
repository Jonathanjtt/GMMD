"""configs/*.toml loading: the shipped experiment config, variants, overrides."""

import pytest

from gmmd import config as C


def test_shipped_config_loads_and_is_complete():
    cfg = C.load_config("conditional_multimodality")
    for section in ("dataset", "teacher", "transition", "features", "diagnostics", "output"):
        assert section in cfg
    tr = cfg["transition"]
    assert 0 < tr["s"] <= tr["s_near"] < tr["t"] <= 1
    assert cfg["teacher"]["predictor"] in ("reverse_diffusion", "euler_maruyama")
    assert cfg["teacher"]["corrector"] in ("none", "langevin")
    assert cfg["name"] == "conditional_multimodality"
    assert cfg["ibw"]["enabled"] is False


def test_variant_inherits_and_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CONFIG_DIR", tmp_path)
    (tmp_path / "base.toml").write_text('[transition]\nt = 0.6\ns = 0.4\n[teacher]\nbatch_size = 8\n')
    (tmp_path / "grp").mkdir()
    (tmp_path / "grp" / "long.toml").write_text('inherit = "base"\n[transition]\ns = 0.2\n')
    cfg = C.load_config("grp/long", {"teacher": {"batch_size": 2}})
    assert cfg["transition"] == {"t": 0.6, "s": 0.2}
    assert cfg["teacher"]["batch_size"] == 2 and cfg["name"] == "grp/long"
    assert "inherit" not in cfg
    with pytest.raises(FileNotFoundError):
        C.load_config("missing")
    (tmp_path / "x.toml").write_text('inherit = "y"\n')
    (tmp_path / "y.toml").write_text('inherit = "x"\n')
    with pytest.raises(ValueError, match="cycle"):
        C.load_config("x")


def test_parse_overrides():
    o = C.parse_overrides(["transition.n_samples=8", "features.extractor=pixel", "umap.enabled=false"])
    assert o == {"transition": {"n_samples": 8}, "features": {"extractor": "pixel"}, "umap": {"enabled": False}}
    with pytest.raises(ValueError):
        C.parse_overrides(["novalue"])
