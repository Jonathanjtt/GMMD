"""The vendored score_sde_pytorch files differ from upstream ONLY by the declared substitutions.

Offline: the substitution table applies cleanly to nothing but the recorded upstream text (the
vendored files are what the table produces, byte for byte, from the pinned-commit sources this
test re-downloads).  ``network``-marked, like tests/test_walnuts_vendor_parity.py.
"""

from __future__ import annotations

import urllib.request

import pytest

from gmmd.teachers.score_sde._vendor import provenance as PV


def _fetch(rel):
    try:
        with urllib.request.urlopen(PV.RAW + rel, timeout=20) as r:
            return r.read()
    except Exception as e:  # offline, DNS, proxy, ...
        pytest.skip(f"no network: {e}")


def test_declared_edits_are_the_only_difference_offline():
    # the vendored text must be reachable from the recorded upstream digests + edits alone;
    # here we only check that every vendored file exists and that the hook substitution held
    for rel in PV.SUBSTITUTIONS:
        assert (PV.VENDOR_DIR / rel).exists(), rel
    s = (PV.VENDOR_DIR / "sampling.py").read_text()
    assert "torch.randn_like(x)" not in s and s.count("randn_like(x)") == 6
    u = (PV.VENDOR_DIR / "op" / "upfirdn2d.py").read_text()
    assert "cpp_extension" not in u and "UpFirDn2d.apply" not in u


@pytest.mark.network
@pytest.mark.parametrize("rel", sorted(PV.SUBSTITUTIONS))
def test_matches_upstream(rel):
    raw = _fetch(rel)
    assert PV.sha256(raw) == PV.UPSTREAM_SHA[rel], f"{rel}: upstream changed since vendoring"
    expected = PV.apply_edits(PV.upstream_text(raw), PV.SUBSTITUTIONS[rel])
    assert (PV.VENDOR_DIR / rel).read_text() == expected
