"""Tests for the revision-2 analyses.

These cover the three things a reviewer re-running the archive would want
checked mechanically:

  1. the statistical helpers behave as documented (Holm ordering, the
     Pitman-Morgan variance test, paired Cohen's d);
  2. the weight functions the manuscript prints in Table 2 agree with the
     implementation the sweep actually used (``weights.py``);
  3. the released scripts reproduce the specific numbers the manuscript
     reports in Section 5.3, Table 9 and Section 6.4.

Item 3 is a regression test on the paper: if the archived per-seed outputs
or the analysis code ever change, the values printed in the manuscript stop
matching and these tests fail.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

# The scripts live in revision2_analyses/ in the released repository and at the
# top level in the manuscript folder; look in both plus revision2_runs/.
SEARCH = [ROOT / "revision2_analyses", ROOT, ROOT / "revision2_runs"]


def _find(filename: str):
    for d in SEARCH:
        p = d / filename
        if p.exists():
            return p
    return None


def _load(name: str, filename: str):
    path = _find(filename)
    if path is None:
        pytest.skip(f"{filename} not present in this checkout")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def vpt():
    return _load("vpt", "variance_pertype_tests.py")


@pytest.fixture(scope="module")
def wm():
    return _load("wm", "weighting_mass.py")


@pytest.fixture(scope="module")
def ar2():
    return _load("ar2", "analyze_revision2.py")


@pytest.fixture(scope="module")
def weights_mod():
    """weights.py ships with the code artifact, not with the manuscript."""
    path = _find("weights.py")
    if path is None:
        pytest.skip("weights.py not present in this checkout")
    return _load("weights_impl", "weights.py")


# ── statistical helpers ───────────────────────────────────────────────────

def test_holm_is_monotone_and_bounded(vpt):
    raw = [0.001, 0.02, 0.04, 0.5]
    adj = vpt.holm(raw)
    assert adj == sorted(adj), "Holm output must be monotone in the raw order"
    assert all(a >= r for a, r in zip(adj, raw)), "adjusted >= raw"
    assert all(a <= 1.0 for a in adj)
    # the smallest p is multiplied by the full family size
    assert adj[0] == pytest.approx(len(raw) * raw[0])


def test_holm_preserves_input_order(vpt):
    raw = [0.5, 0.001, 0.04]
    adj = vpt.holm(raw)
    assert adj[1] == pytest.approx(3 * 0.001)
    assert adj[0] >= adj[2]


def test_holm_matches_between_implementations(vpt, ar2):
    raw = [0.0019, 0.0019, 0.03, 0.44, 0.9]
    a = vpt.holm(raw)
    b = ar2.holm(raw)
    assert a == pytest.approx(b)


def test_pitman_morgan_detects_no_difference(vpt):
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 1.0, 200)
    other = base + rng.normal(0.0, 0.01, 200)
    _, p, _ = vpt.pitman_morgan(base, other)
    assert p > 0.05, "equal variances must not be flagged"


def test_pitman_morgan_detects_a_large_ratio(vpt):
    rng = np.random.default_rng(1)
    common = rng.normal(0.0, 1.0, 60)
    wide = common * 5.0
    _, p, _ = vpt.pitman_morgan(wide, common)
    assert p < 0.001, "a 5x dispersion ratio must be flagged"


def test_pitman_morgan_is_symmetric_in_p(vpt):
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 40)
    b = a * 2.0
    _, p1, _ = vpt.pitman_morgan(a, b)
    _, p2, _ = vpt.pitman_morgan(b, a)
    assert p1 == pytest.approx(p2)


def test_pitman_morgan_rejects_unpaired_input(vpt):
    with pytest.raises(ValueError):
        vpt.pitman_morgan([1.0, 2.0, 3.0], [1.0, 2.0])


def test_paired_cohens_d_sign_and_scale(vpt):
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = a - 1.0
    assert vpt.cohens_d_paired(a, b) == float("inf")   # zero paired variance
    c = np.array([1.0, 3.0, 2.0, 5.0])
    d = vpt.cohens_d_paired(c, b)
    assert d > 0
    assert vpt.cohens_d_paired(b, c) == pytest.approx(-d)


# ── Table 2 formulas vs the implementation used by the sweep ──────────────

def test_manuscript_formulas_match_weights_py(wm, weights_mod):
    r = np.array([0.0, 0.05, 0.1, 0.19, 0.2, 0.35, 0.49, 0.5, 0.75, 1.0])
    y = np.ones_like(r, dtype=int)
    for name, f in wm.FORMULAS.items():
        got = weights_mod.weights_for(name, y, r)
        want = np.array([f(x) for x in r])
        assert got == pytest.approx(want, abs=1e-12), name


def test_benign_samples_are_never_reweighted(weights_mod):
    r = np.array([0.0, 0.0, 0.9])
    y = np.array([0, 0, 1])
    for name in ("linear", "sqrt", "log", "binary_high", "step_3level"):
        w = weights_mod.weights_for(name, y, r)
        assert w[0] == 1.0 and w[1] == 1.0, name


def test_weight_functions_are_bounded_on_the_unit_interval(wm):
    for name, f in wm.FORMULAS.items():
        vals = [f(x) for x in np.linspace(0.0, 1.0, 101)]
        assert min(vals) >= 0.0, name
        assert max(vals) <= 1.0 + 1e-12, name
        assert f(1.0) == pytest.approx(1.0), name


def test_suppression_factors_reported_in_section_6_4(wm):
    """Section 6.4 prints 2.00x, 2.50x, 3.16x, 3.46x and 10.00x."""
    expected = {"binary_high": 2.00, "step_3level": 2.50, "sqrt": 3.16,
                "log": 3.46, "linear": 10.00}
    for name, want in expected.items():
        f = wm.FORMULAS[name]
        assert f(1.0) / f(0.10) == pytest.approx(want, abs=0.005), name


def test_neither_summary_orders_the_measured_damage(wm):
    """The negative claim in Section 6.4 must keep holding."""
    diag = wm.json.loads(wm._find_mechanism().read_text(encoding="utf-8"))
    base = next(d for d in diag["diagnostics"]
                if abs(d["T"] - wm.BASELINE_T) < 1e-9)
    hist = base["detection_ratio_histogram_10bins"]
    mids = [(a + b) / 2 for a, b in zip(hist["edges"][:-1], hist["edges"][1:])]
    total = sum(hist["counts"])

    def mass_removed(f):
        return 1.0 - sum(c * f(m) for c, m in zip(hist["counts"], mids)) / total

    by_supp = sorted(wm.FORMULAS,
                     key=lambda n: wm.FORMULAS[n](1.0) / wm.FORMULAS[n](0.10))
    by_mass = sorted(wm.FORMULAS, key=lambda n: mass_removed(wm.FORMULAS[n]))
    by_harm = sorted(wm.FORMULAS, key=lambda n: -wm.LGBM_DELTA_PP[n])
    assert by_supp != by_harm
    assert by_mass != by_harm
    # ...while the two regularities the manuscript does claim must hold
    smooth = ["sqrt", "log", "linear"]
    assert (sorted(smooth, key=lambda n: wm.FORMULAS[n](1.0) / wm.FORMULAS[n](0.10))
            == sorted(smooth, key=lambda n: -wm.LGBM_DELTA_PP[n]))
    assert (max(wm.LGBM_DELTA_PP[n] for n in ("binary_high", "step_3level"))
            < max(wm.LGBM_DELTA_PP[n] for n in ("sqrt", "log")))


# ── regression on the numbers the manuscript prints ───────────────────────

@pytest.fixture(scope="module")
def results(vpt):
    d = vpt._find_results()
    if not (d / "lgbm_fine" / "summary.json").exists():
        pytest.skip("results_aggregated/ not present in this checkout")
    return d


def test_section_5_3_variance_ratios_and_p_values(vpt, results):
    """Section 5.3 lists 5.7x, 7.9x, 7.4x, 7.1x and 6.6x, all p_Holm < 1e-3."""
    ve = vpt.variance_equality(results)
    by_T = {round(r["T"], 3): r for r in ve["rows"]}
    expected = {0.08: 5.7, 0.10: 7.9, 0.12: 7.4, 0.15: 7.1, 0.18: 6.6}
    for T, ratio in expected.items():
        row = by_T[T]
        assert row["sd_ratio"] == pytest.approx(ratio, abs=0.05), T
        assert row["p_holm"] < 1e-3, T
    # and the boundary claim: nothing in 0.20-0.40 is significant
    for T in (0.20, 0.25, 0.30, 0.40):
        assert by_T[T]["p_holm"] >= 0.05, T
    # T = 0.50 does pass, which the manuscript reports explicitly
    assert by_T[0.50]["p_holm"] < 0.05
    assert by_T[0.50]["sd_ratio"] == pytest.approx(4.5, abs=0.1)


def test_table_9_per_file_type_values(vpt, results):
    pt = vpt.per_file_type(results)
    # baselines
    assert pt["win32"]["baseline"]["mean"] * 100 == pytest.approx(68.43, abs=0.01)
    assert pt["win64"]["baseline"]["mean"] * 100 == pytest.approx(56.74, abs=0.01)
    assert pt["dot_net"]["baseline"]["mean"] * 100 == pytest.approx(78.59, abs=0.01)
    # the Win64 divergence that changed the manuscript's conclusion
    win64 = {round(r["T"], 3): r for r in pt["win64"]["rows"]}
    assert win64[0.15]["delta_pp"] == pytest.approx(-1.06, abs=0.01)
    assert win64[0.15]["p_holm"] == pytest.approx(0.0078, abs=0.0005)
    assert win64[0.15]["n_seeds_favoring"] == 0
    assert win64[0.10]["p_holm"] == pytest.approx(0.0391, abs=0.0005)
    # while Win32 and .NET still gain at T = 0.15
    for ft, delta in (("win32", 1.61), ("dot_net", 0.80)):
        row = {round(r["T"], 3): r for r in pt[ft]["rows"]}[0.15]
        assert row["delta_pp"] == pytest.approx(delta, abs=0.01), ft
        assert row["n_seeds_favoring"] == 10, ft
        assert row["p_holm"] == pytest.approx(0.0059, abs=0.0005), ft


def test_exact_wilcoxon_floor_is_respected(vpt, results):
    """No reported p may sit below the exact two-sided floor 2/2**10."""
    floor = 2 / 2 ** 10
    pt = vpt.per_file_type(results)
    for ft, blk in pt.items():
        for row in blk["rows"]:
            assert row["p_raw"] >= floor - 1e-12, (ft, row["T"])
            assert row["p_holm"] >= floor - 1e-12, (ft, row["T"])


def test_per_subtype_friedman_values(vpt, results):
    pt = vpt.per_file_type(results)
    expected = {"win32": 24.60, "win64": 25.62, "dot_net": 24.06}
    for ft, chi2 in expected.items():
        assert pt[ft]["friedman_chi2"] == pytest.approx(chi2, abs=0.02), ft
        assert pt[ft]["friedman_p"] < 3e-5, ft
