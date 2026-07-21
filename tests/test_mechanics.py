import cv2
import numpy as np

from analysis.mechanics import (
    adaptive_segmentation,
    boundary_from_mask,
    boundary_overlay,
    contour_curvature,
    contour_metrics,
    crack_metrics,
    curl_map,
    displacement_field,
    model_segmentation,
    normalized_shape_change,
    segmentation_qc,
    shape_descriptors,
    strain_tensor,
    track_crack_tips,
)
from core import SegmentationConfig


def rectangle_frame():
    frame = np.zeros((128, 128), dtype=np.uint16)
    cv2.rectangle(frame, (30, 35), (95, 100), 4000, -1)
    return frame


def test_adaptive_segmentation_exports_mask_and_confidence():
    mask, confidence = adaptive_segmentation(
        rectangle_frame(),
        SegmentationConfig(method="adaptive", adaptive_c=5, minimum_segment_area=100),
    )
    assert mask.dtype == np.uint8
    assert confidence.dtype == np.float32
    assert 3500 < np.count_nonzero(mask) < 5000
    assert np.all((confidence >= 0) & (confidence <= 1))


def test_model_plugin_contract_accepts_mask_and_confidence():
    frame = rectangle_frame()

    def plugin(data):
        return data > 100, np.full(data.shape, 0.8, dtype=np.float32)

    mask, confidence = model_segmentation(
        frame, SegmentationConfig(method="model", model_plugin="test:plugin"), plugin
    )
    assert np.count_nonzero(mask) > 4000
    assert np.isclose(confidence.mean(), 0.8)


def test_boundary_comes_from_mask_and_overlay_is_red():
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(mask, (32, 32), 15, 255, -1)
    boundary, used_fallback = boundary_from_mask(mask)
    overlay = boundary_overlay(mask, boundary)
    assert not used_fallback
    assert 70 < np.count_nonzero(boundary) < 120
    assert np.all(overlay[boundary > 0] == np.array([255, 0, 0]))


def test_boundary_uses_canny_only_for_empty_mask():
    frame = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(frame, (15, 15), (45, 45), 255, -1)
    boundary, used_fallback = boundary_from_mask(np.zeros_like(frame), frame, 20, 80)
    assert used_fallback
    assert np.count_nonzero(boundary) > 0


def test_curvature_and_shape_descriptors_are_finite():
    mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(mask, (64, 64), 30, 255, -1)
    signed_map, curvature = contour_curvature(mask)
    shape = shape_descriptors(mask)
    assert np.count_nonzero(signed_map) > 0
    assert curvature["mean_absolute_curvature"] > 0
    assert curvature["max_absolute_curvature"] >= curvature["mean_absolute_curvature"]
    assert 0.85 < shape["circularity"] <= 1.05
    assert 0.9 < shape["elongation"] < 1.1


def test_contours_are_extracted_and_arc_lengths_measured():
    mask = np.zeros((100, 120), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (30, 30), 255, -1)
    cv2.rectangle(mask, (60, 20), (90, 50), 255, -1)
    contours, metrics = contour_metrics(mask)
    assert len(contours) == 2
    assert metrics["contour_count"] == 2
    assert np.isclose(metrics["total_contour_length"], 200, atol=2)
    assert np.isclose(metrics["mean_contour_length"], 100, atol=2)


def test_normalized_shape_change_uses_first_frame_baseline():
    records = [
        {"area": 10, "perimeter": 8, "circularity": 0.5, "elongation": 2, "angle": 170},
        {"area": 15, "perimeter": 10, "circularity": 0.6, "elongation": 3, "angle": 10},
    ]
    change = normalized_shape_change(records)
    assert np.allclose(change["area"], [0, 0.5])
    assert np.allclose(change["elongation"], [0, 0.5])
    assert np.isclose(change["angle"][1], 20 / 180)


def test_registration_exports_known_displacement_field():
    before = np.zeros((96, 96), dtype=np.uint8)
    cv2.circle(before, (40, 42), 12, 255, -1)
    matrix = np.float32([[1, 0, 4], [0, 1, 3]])
    after = cv2.warpAffine(before, matrix, (96, 96))
    field = displacement_field(before, after, method="registration")
    assert field.shape == (96, 96, 2)
    assert np.allclose(field.mean(axis=(0, 1)), [4, 3], atol=0.15)


def test_signed_curl_and_small_strain_maps():
    y, x = np.mgrid[:40, :50]
    rotational = np.stack([-y, x], axis=-1).astype(np.float32)
    assert np.allclose(curl_map(rotational)[2:-2, 2:-2], 2.0)

    displacement = np.stack([0.1 * x + 0.1 * y, 0.2 * x + 0.2 * y], axis=-1)
    exx, eyy, exy = strain_tensor(displacement, finite=False)
    assert np.allclose(exx, 0.1)
    assert np.allclose(eyy, 0.2)
    assert np.allclose(exy, 0.15)


def test_finite_strain_includes_quadratic_gradient_terms():
    y, x = np.mgrid[:30, :30]
    displacement = np.stack([0.2 * x, np.zeros_like(x)], axis=-1)
    exx, eyy, exy = strain_tensor(displacement, finite=True)
    assert np.allclose(exx, 0.22)
    assert np.allclose(eyy, 0.0)
    assert np.allclose(exy, 0.0)


def test_crack_skeleton_length_branches_and_tip_tracking():
    line = np.zeros((64, 64), dtype=np.uint8)
    cv2.line(line, (10, 30), (50, 30), 255, 5)
    skeleton, metrics = crack_metrics(line)
    assert np.count_nonzero(skeleton) > 0
    assert 38 <= metrics["length"] <= 42
    assert metrics["branch_count"] == 0
    assert metrics["tip_count"] == 2
    assert 4.5 <= metrics["mean_width"] <= 8.0
    assert metrics["max_width"] >= metrics["mean_width"]
    assert np.count_nonzero(metrics["width_map"]) == np.count_nonzero(skeleton)

    cross = line.copy()
    cv2.line(cross, (30, 10), (30, 50), 255, 5)
    _, cross_metrics = crack_metrics(cross)
    assert cross_metrics["branch_count"] == 1
    assert cross_metrics["tip_count"] == 4

    tracked = track_crack_tips([(0, 0), (10, 0)], [(3, 4), (13, 4)])
    assert np.isclose(tracked["mean_tip_displacement"], 5.0)


def test_segmentation_qc_penalizes_border_contact_and_fallback():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:10, :10] = 255
    confidence = np.full(mask.shape, 0.8, dtype=np.float32)
    qc = segmentation_qc(mask, confidence, used_canny_fallback=True)
    assert np.isclose(qc["confidence"], 0.8)
    assert qc["touches_border"] == 1
    assert qc["canny_fallback"] == 1
    assert qc["qc_score"] < qc["confidence"]
