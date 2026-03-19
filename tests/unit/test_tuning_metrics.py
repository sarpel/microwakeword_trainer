"""Unit tests for src.tuning.metrics — RED phase (all tests must FAIL with ImportError)."""

from __future__ import annotations

import numpy as np

# These imports will ALL FAIL (ImportError) until metrics.py is created — that's the RED phase.
from src.tuning.metrics import (
    ErrorMemory,
    ParetoArchive,
    ThresholdOptimizer,
    TuneMetrics,
    apply_temperature,
    compute_ece,
    compute_hypervolume,
    fit_temperature,
)

# ---------------------------------------------------------------------------
# TuneMetrics tests
# ---------------------------------------------------------------------------


def test_tune_metrics_default_construction() -> None:
    m = TuneMetrics()
    assert m.fah == float("inf")
    assert m.recall == 0.0
    assert m.auc_roc == 0.0
    assert m.auc_pr == 0.0
    assert m.ece == 1.0
    assert m.threshold == 0.5
    assert m.threshold_uint8 == 128
    assert m.precision == 0.0
    assert m.f1 == 0.0
    assert m.total_positives == 0
    assert m.total_negatives == 0


def test_tune_metrics_construction_with_values() -> None:
    m = TuneMetrics(
        fah=0.3,
        recall=0.95,
        auc_roc=0.99,
        auc_pr=0.98,
        ece=0.02,
        threshold=0.42,
        threshold_uint8=107,
        precision=0.95,
        f1=0.95,
        total_positives=100,
        total_negatives=1000,
        false_positives=3,
        false_negatives=5,
        ambient_duration_hours=2.5,
    )
    assert m.fah == 0.3
    assert m.recall == 0.95
    assert m.threshold_uint8 == 107
    assert m.ambient_duration_hours == 2.5


def test_tune_metrics_dominates_strict() -> None:
    a = TuneMetrics(fah=0.3, recall=0.95, auc_pr=0.98)
    b = TuneMetrics(fah=0.5, recall=0.90, auc_pr=0.93)
    assert a.dominates(b), "a should dominate b (better in all 3 objectives)"
    assert not b.dominates(a)


def test_tune_metrics_dominates_equal_not_dominating() -> None:
    a = TuneMetrics(fah=0.3, recall=0.95, auc_pr=0.98)
    assert not a.dominates(a), "a should not dominate itself (not strictly better)"


def test_tune_metrics_dominates_non_dominated_pair() -> None:
    a = TuneMetrics(fah=0.2, recall=0.90, auc_pr=0.95)  # lower fah
    b = TuneMetrics(fah=0.5, recall=0.97, auc_pr=0.96)  # higher recall
    assert not a.dominates(b), "a does not dominate b (recall worse)"
    assert not b.dominates(a), "b does not dominate a (fah worse)"


def test_tune_metrics_meets_target_true() -> None:
    m = TuneMetrics(fah=0.2, recall=0.96)
    assert m.meets_target(target_fah=0.5, target_recall=0.90)


def test_tune_metrics_meets_target_false_fah() -> None:
    m = TuneMetrics(fah=1.5, recall=0.96)
    assert not m.meets_target(target_fah=0.5, target_recall=0.90)


def test_tune_metrics_meets_target_false_recall() -> None:
    m = TuneMetrics(fah=0.2, recall=0.80)
    assert not m.meets_target(target_fah=0.5, target_recall=0.90)


def test_tune_metrics_to_dict_has_required_keys() -> None:
    m = TuneMetrics(fah=0.3, recall=0.95, auc_roc=0.99, auc_pr=0.98, ece=0.02, threshold=0.42, threshold_uint8=107, precision=0.94, f1=0.945)
    d = m.to_dict()
    for key in ("fah", "recall", "auc_roc", "auc_pr", "ece", "threshold", "threshold_uint8", "precision", "f1"):
        assert key in d, f"Missing key: {key}"
    assert d["fah"] == 0.3
    assert d["threshold_uint8"] == 107


# ---------------------------------------------------------------------------
# ParetoArchive tests (NEW API: try_add(metrics, candidate_id))
# ---------------------------------------------------------------------------


def test_pareto_archive_add_single() -> None:
    archive = ParetoArchive(max_size=10)
    m = TuneMetrics(fah=0.5, recall=0.90, auc_pr=0.92)
    result = archive.try_add(m, candidate_id="c1")
    assert result is True
    assert len(archive) == 1


def test_pareto_archive_two_non_dominated() -> None:
    archive = ParetoArchive(max_size=10)
    # m1: low FAH but low recall — m2: high FAH but high recall — neither dominates the other
    m1 = TuneMetrics(fah=1.0, recall=0.85, auc_pr=0.90)
    m2 = TuneMetrics(fah=2.0, recall=0.95, auc_pr=0.98)
    archive.try_add(m1, "c1")
    archive.try_add(m2, "c2")
    assert len(archive) == 2, "Both non-dominated points should be kept"


def test_pareto_archive_dominated_rejected() -> None:
    archive = ParetoArchive(max_size=10, diversity_threshold=0.01)
    m_good = TuneMetrics(fah=0.3, recall=0.95, auc_pr=0.98)
    m_bad = TuneMetrics(fah=0.5, recall=0.90, auc_pr=0.93)  # dominated
    archive.try_add(m_good, "c_good")
    result = archive.try_add(m_bad, "c_bad")
    assert result is False


def test_pareto_archive_replaces_dominated_by_new() -> None:
    archive = ParetoArchive(max_size=10, diversity_threshold=0.01)
    m_old = TuneMetrics(fah=0.5, recall=0.90, auc_pr=0.93)
    m_new = TuneMetrics(fah=0.3, recall=0.95, auc_pr=0.98)  # dominates m_old
    archive.try_add(m_old, "old")
    archive.try_add(m_new, "new")
    assert len(archive) == 1


def test_pareto_archive_respects_max_size() -> None:
    archive = ParetoArchive(max_size=3, diversity_threshold=0.0)
    for i in range(10):
        m = TuneMetrics(fah=float(i + 1), recall=1.0 / (i + 1) + 0.5, auc_pr=0.9)
        archive.try_add(m, f"c{i}")
    assert len(archive) <= 3


def test_pareto_archive_get_best_meeting_targets() -> None:
    archive = ParetoArchive(max_size=10)
    archive.try_add(TuneMetrics(fah=0.4, recall=0.91, auc_pr=0.82), "c1")
    archive.try_add(TuneMetrics(fah=0.2, recall=0.89, auc_pr=0.83), "c2")
    best = archive.get_best(target_fah=0.5, target_recall=0.90)
    assert best is not None
    # c1 meets both targets (fah=0.4 <= 0.5, recall=0.91 >= 0.90)
    assert best[1] == "c1" or best[0].recall >= 0.90  # flexible: return (metrics, id) or just metrics


def test_pareto_archive_get_frontier_points_structure() -> None:
    archive = ParetoArchive(max_size=10)
    archive.try_add(TuneMetrics(fah=0.3, recall=0.95, auc_pr=0.98), "c1")
    archive.try_add(TuneMetrics(fah=0.5, recall=0.92, auc_pr=0.94), "c2")
    points = archive.get_frontier_points()
    assert isinstance(points, list)
    assert len(points) >= 1
    for p in points:
        assert "fah" in p
        assert "recall" in p
        assert "auc_pr" in p


def test_pareto_archive_empty_len() -> None:
    archive = ParetoArchive(max_size=10)
    assert len(archive) == 0


# ---------------------------------------------------------------------------
# ErrorMemory tests
# ---------------------------------------------------------------------------


def test_error_memory_update_and_persistent_fa() -> None:
    em = ErrorMemory()
    indices = np.array([0, 1, 2])
    y_true = np.array([0.0, 0.0, 1.0])
    y_pred = np.array([0.9, 0.8, 0.1])  # idx 0,1 = FA; idx 2 = miss
    for _ in range(4):  # repeat 4 times to exceed min_count=3
        em.update(indices, y_true, y_pred, threshold=0.5)
    fa_indices = em.get_persistent_fa_indices(min_count=3)
    assert 0 in fa_indices
    assert 1 in fa_indices
    assert 2 not in fa_indices


def test_error_memory_persistent_miss() -> None:
    em = ErrorMemory()
    indices = np.array([5, 6])
    y_true = np.array([1.0, 1.0])
    y_pred = np.array([0.1, 0.2])  # both misses
    for _ in range(4):
        em.update(indices, y_true, y_pred, threshold=0.5)
    miss_indices = em.get_persistent_miss_indices(min_count=3)
    assert 5 in miss_indices
    assert 6 in miss_indices


def test_error_memory_near_boundary() -> None:
    em = ErrorMemory()
    indices = np.array([10])
    y_true = np.array([0.0])
    y_pred = np.array([0.51])  # near threshold 0.5
    for _ in range(3):
        em.update(indices, y_true, y_pred, threshold=0.5)
    near = em.get_near_boundary_indices(threshold=0.5, margin=0.1)
    assert 10 in near


# ---------------------------------------------------------------------------
# Temperature / Calibration tests
# ---------------------------------------------------------------------------


def test_compute_ece_perfect_calibration() -> None:
    # If predictions match ground truth, ECE should be near 0
    y_true = np.array([1.0, 1.0, 0.0, 0.0])
    y_prob = np.array([0.95, 0.90, 0.05, 0.10])
    ece = compute_ece(y_true, y_prob)
    assert ece >= 0.0
    assert ece < 0.2


def test_compute_ece_empty_returns_zero() -> None:
    ece = compute_ece(np.array([]), np.array([]))
    assert ece == 0.0


def test_apply_temperature_identity_at_one() -> None:
    scores = np.array([0.1, 0.5, 0.9])
    result = apply_temperature(scores, 1.0)
    np.testing.assert_allclose(result, scores, atol=1e-5)


def test_apply_temperature_sharpens_at_low_t() -> None:
    scores = np.array([0.6])  # above 0.5
    sharpened = apply_temperature(scores, 0.5)  # low temp sharpens
    assert sharpened[0] > scores[0], "Low temperature should push scores toward extremes"


def test_fit_temperature_returns_positive_float() -> None:
    np.random.seed(42)
    y_true = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    probs = np.array([0.8, 0.7, 0.75, 0.3, 0.25, 0.2])
    t = fit_temperature(probs, y_true)
    assert isinstance(t, float)
    assert t > 0.0


# ---------------------------------------------------------------------------
# ThresholdOptimizer tests
# ---------------------------------------------------------------------------


def test_threshold_optimizer_instantiation() -> None:
    opt = ThresholdOptimizer()
    assert opt is not None


def test_threshold_optimizer_optimize_basic() -> None:
    opt = ThresholdOptimizer()
    np.random.seed(0)
    # 100 positives at high scores, 900 negatives at low scores
    y_true = np.concatenate([np.ones(100), np.zeros(900)])
    y_scores = np.concatenate([np.random.uniform(0.7, 1.0, 100), np.random.uniform(0.0, 0.3, 900)])
    result = opt.optimize(
        y_true=y_true,
        y_scores=y_scores,
        ambient_duration_hours=1.0,
        target_fah=0.5,
        target_recall=0.90,
    )
    assert len(result) == 3  # (threshold_float32, threshold_uint8, TuneMetrics)
    threshold_f, threshold_u8, metrics = result
    assert 0.0 <= threshold_f <= 1.0
    assert 0 <= threshold_u8 <= 255
    assert isinstance(metrics, TuneMetrics)


def test_threshold_optimizer_empty_arrays() -> None:
    opt = ThresholdOptimizer()
    result = opt.optimize(
        y_true=np.array([]),
        y_scores=np.array([]),
        ambient_duration_hours=1.0,
        target_fah=0.5,
        target_recall=0.90,
    )
    threshold_f, threshold_u8, metrics = result
    assert threshold_f == 0.5  # should use default


# ---------------------------------------------------------------------------
# compute_hypervolume tests
# ---------------------------------------------------------------------------


def test_hypervolume_empty_returns_zero() -> None:
    hv = compute_hypervolume([], reference=(10.0, 0.0))
    assert hv == 0.0


def test_hypervolume_single_point_positive() -> None:
    hv = compute_hypervolume([(1.0, 0.95)], reference=(10.0, 0.0))
    assert hv > 0.0


def test_hypervolume_known_pareto_front() -> None:
    points = [(1.0, 0.95), (2.0, 0.90), (3.0, 0.80)]
    hv = compute_hypervolume(points, reference=(10.0, 0.0))
    assert hv > 0.0, "Hypervolume of non-empty Pareto front must be positive"


def test_hypervolume_larger_front_greater_hv() -> None:
    """More Pareto points covering more space should give larger hypervolume."""
    points_small = [(1.0, 0.95)]
    points_large = [(1.0, 0.95), (2.0, 0.85), (3.0, 0.75)]
    hv_small = compute_hypervolume(points_small, reference=(10.0, 0.0))
    hv_large = compute_hypervolume(points_large, reference=(10.0, 0.0))
    assert hv_large >= hv_small


def test_hypervolume_reference_affects_result() -> None:
    points = [(1.0, 0.95)]
    hv_narrow = compute_hypervolume(points, reference=(5.0, 0.0))
    hv_wide = compute_hypervolume(points, reference=(10.0, 0.0))
    assert hv_wide >= hv_narrow, "Wider reference should give >= hypervolume"
