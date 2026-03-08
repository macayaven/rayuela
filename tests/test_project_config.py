from __future__ import annotations

import numpy as np
import pytest

import project_config


def test_distance_metric_signs() -> None:
    assert project_config.DistanceMetric.EUCLIDEAN.sign == -1
    assert project_config.DistanceMetric.COSINE.sign == 1


def test_z_score_handles_metric_direction() -> None:
    null_dist = np.array([1.0, 2.0, 3.0], dtype=float)

    cosine = project_config.z_score(3.0, null_dist, project_config.DistanceMetric.COSINE)
    euclidean = project_config.z_score(1.0, null_dist, project_config.DistanceMetric.EUCLIDEAN)

    assert cosine > 0
    assert euclidean > 0


def test_z_score_rejects_non_enum_metric() -> None:
    with pytest.raises(TypeError):
        project_config.z_score(1.0, np.array([1.0]), "cosine")  # type: ignore[arg-type]


def test_z_score_returns_zero_when_null_distribution_has_no_variance() -> None:
    null_dist = np.array([1.0, 1.0, 1.0], dtype=float)
    assert project_config.z_score(1.0, null_dist, project_config.DistanceMetric.COSINE) == 0.0


def test_z_standardize_leaves_zero_variance_columns_stable() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]])
    standardized = project_config.z_standardize(matrix)

    assert np.allclose(standardized[:, 0].mean(), 0.0)
    assert np.allclose(standardized[:, 1], np.zeros(3))


def test_z_standardize_scores_dict_returns_lookup_vectors() -> None:
    scores = {
        1: {"a": 1.0, "b": 5.0},
        2: {"a": 3.0, "b": 5.0},
    }

    result = project_config.z_standardize_scores_dict(scores, ["a", "b"])

    assert set(result) == {1, 2}
    assert result[1].shape == (2,)
    assert result[1][1] == 0.0


def test_continuity_corrected_percentile_tracks_extreme_count() -> None:
    null_dist = np.array([0.1, 0.2, 0.3])
    percentile = project_config.continuity_corrected_percentile(
        0.25,
        null_dist,
        project_config.DistanceMetric.COSINE,
    )
    assert percentile == pytest.approx(75.0)


def test_continuity_corrected_percentile_handles_euclidean_and_type_errors() -> None:
    null_dist = np.array([1.0, 2.0, 3.0])
    percentile = project_config.continuity_corrected_percentile(
        1.5,
        null_dist,
        project_config.DistanceMetric.EUCLIDEAN,
    )

    assert percentile == pytest.approx(75.0)
    with pytest.raises(TypeError):
        project_config.continuity_corrected_percentile(1.0, null_dist, "euclidean")  # type: ignore[arg-type]


def test_get_reading_paths_and_all_chapters_are_loaded() -> None:
    tablero, linear = project_config.get_reading_paths()

    assert linear == list(range(1, 57))
    assert tablero[0] == 73
    assert project_config.get_all_chapters()[-1] == 155


def test_filter_excluded_dims_removes_temporal_clarity_column() -> None:
    matrix = np.arange(40, dtype=float).reshape(2, 20)
    filtered = project_config.filter_excluded_dims(matrix)

    assert filtered.shape == (2, 19)
    excluded_index = project_config.DIMS_ORDERED_ALL.index("temporal_clarity")
    assert matrix[0, excluded_index] not in filtered[0]
