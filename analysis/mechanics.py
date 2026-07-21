"""Interpretable, independently testable mechanics computations.

The functions in this module do not perform file I/O.  They accept arrays and
return arrays or plain dictionaries so every metric can be unit tested without
running the BARCODE GUI or pipeline.
"""

from __future__ import annotations

import heapq
import importlib
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
from skimage.morphology import skeletonize

from core import SegmentationConfig


ModelPlugin = Callable[[np.ndarray], Any]


def as_uint8(frame: np.ndarray) -> np.ndarray:
    """Robustly map a two-dimensional microscopy frame to 8-bit intensity."""
    data = np.asarray(frame)
    if data.ndim != 2:
        raise ValueError("Mechanics analysis expects a two-dimensional channel frame")
    finite = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if finite.dtype == np.uint8:
        return finite.copy()
    low, high = np.percentile(finite, (0.5, 99.5))
    if high <= low:
        return np.zeros(finite.shape, dtype=np.uint8)
    return np.clip((finite - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def validate_config(config: SegmentationConfig) -> None:
    if config.method not in {"adaptive", "canny", "model"}:
        raise ValueError("Segmentation method must be adaptive, canny, or model")
    if config.adaptive_block_size < 3 or config.adaptive_block_size % 2 == 0:
        raise ValueError("Adaptive block size must be an odd number of at least 3")
    if not 0 <= config.canny_low < config.canny_high:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high")
    if config.blur_kernel < 1 or config.blur_kernel % 2 == 0:
        raise ValueError("Blur kernel must be a positive odd number")
    if min(config.open_kernel, config.close_kernel) < 1:
        raise ValueError("Morphology kernels must be positive")
    if min(config.open_iterations, config.close_iterations, config.minimum_segment_area) < 0:
        raise ValueError("Morphology iterations and minimum area cannot be negative")
    if config.method == "model" and not config.model_plugin:
        raise ValueError("Model segmentation requires a module:function plug-in path")
    if config.deformation_method not in {"flow", "registration"}:
        raise ValueError("Deformation method must be flow or registration")
    if config.strain_type not in {"small", "finite"}:
        raise ValueError("Strain type must be small or finite")


def _remove_small_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for label_id in range(1, count):
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= minimum_area:
            clean[labels == label_id] = 1
    return clean * 255


def adaptive_segmentation(
    frame: np.ndarray, config: SegmentationConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Segment with a local threshold and return mask plus confidence map."""
    validate_config(config)
    gray = as_uint8(frame)
    blurred = cv2.GaussianBlur(gray, (config.blur_kernel, config.blur_kernel), 0)
    threshold_type = cv2.THRESH_BINARY_INV if config.invert_mask else cv2.THRESH_BINARY
    mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        config.adaptive_block_size,
        config.adaptive_c,
    )
    # Gate the local decision with Otsu so uniform background and uniform object
    # interiors do not both become foreground when C is positive.
    _, global_mask = cv2.threshold(
        blurred, 0, 255, threshold_type | cv2.THRESH_OTSU
    )
    mask = cv2.bitwise_and(mask, global_mask)
    if config.open_iterations:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.open_kernel, config.open_kernel)
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, iterations=config.open_iterations
        )
    if config.close_iterations:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.close_kernel, config.close_kernel)
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel, iterations=config.close_iterations
        )
    mask = _remove_small_components(mask, config.minimum_segment_area)

    local_mean = cv2.GaussianBlur(
        blurred.astype(np.float32),
        (config.adaptive_block_size, config.adaptive_block_size),
        0,
    )
    local_square_mean = cv2.GaussianBlur(
        blurred.astype(np.float32) ** 2,
        (config.adaptive_block_size, config.adaptive_block_size),
        0,
    )
    local_std = np.sqrt(np.maximum(local_square_mean - local_mean**2, 0.0))
    signed_margin = blurred.astype(np.float32) - (local_mean - config.adaptive_c)
    if config.invert_mask:
        signed_margin *= -1
    # Confidence is distance from the decision boundary, bounded to [0, 1].
    confidence = np.clip(
        np.abs(signed_margin) / (local_std + abs(config.adaptive_c) + 1e-6),
        0.0,
        1.0,
    ).astype(np.float32)
    return mask, confidence


def load_model_plugin(specification: str) -> ModelPlugin:
    """Load a `module:function` segmentation plug-in."""
    try:
        module_name, function_name = specification.split(":", 1)
    except ValueError as error:
        raise ValueError("Model plug-in must use module:function syntax") from error
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise ValueError(f"Model plug-in is not callable: {specification}")
    return function


def model_segmentation(
    frame: np.ndarray, config: SegmentationConfig, plugin: ModelPlugin | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Run an optional learned plug-in with a stable mask/confidence contract."""
    function = plugin or load_model_plugin(config.model_plugin)
    output = function(frame)
    if isinstance(output, tuple) and len(output) == 2:
        mask, confidence = output
    else:
        mask, confidence = output, None
    mask_array = np.asarray(mask)
    if mask_array.shape != frame.shape:
        raise ValueError("Model plug-in mask must match the input frame shape")
    mask_u8 = _remove_small_components(mask_array, config.minimum_segment_area)
    if confidence is None:
        confidence_array = np.ones(frame.shape, dtype=np.float32)
    else:
        confidence_array = np.asarray(confidence, dtype=np.float32)
        if confidence_array.shape != frame.shape:
            raise ValueError("Model confidence must match the input frame shape")
        confidence_array = np.clip(confidence_array, 0.0, 1.0)
    return mask_u8, confidence_array


def canny_segmentation(
    frame: np.ndarray, config: SegmentationConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible Canny contour filling with binary confidence."""
    gray = as_uint8(frame)
    blurred = cv2.GaussianBlur(gray, (config.blur_kernel, config.blur_kernel), 0)
    edges = cv2.Canny(blurred, config.canny_low, config.canny_high)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.close_kernel, config.close_kernel)
    )
    closed = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, kernel, iterations=config.close_iterations
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(edges)
    contours = [c for c in contours if cv2.contourArea(c) >= config.minimum_segment_area]
    if contours:
        cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
    return mask, (edges > 0).astype(np.float32)


def segment_mask(
    frame: np.ndarray,
    config: SegmentationConfig,
    plugin: ModelPlugin | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch a configured segmentation method."""
    if config.method == "model":
        return model_segmentation(frame, config, plugin)
    if config.method == "canny":
        return canny_segmentation(frame, config)
    return adaptive_segmentation(frame, config)


def boundary_from_mask(
    mask: np.ndarray,
    fallback_frame: np.ndarray | None = None,
    canny_low: int = 50,
    canny_high: int = 150,
) -> tuple[np.ndarray, bool]:
    """Extract a one-pixel mask contour, with Canny only when the mask is empty."""
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = np.zeros_like(binary)
    if contours:
        cv2.drawContours(boundary, contours, -1, 255, 1)
        return boundary, False
    if fallback_frame is None:
        return boundary, False
    return cv2.Canny(as_uint8(fallback_frame), canny_low, canny_high), True


def boundary_overlay(frame: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Return an RGB evidence image with boundaries shown in red."""
    gray = as_uint8(frame)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    overlay[np.asarray(boundary) > 0] = (255, 0, 0)
    return overlay


def segmentation_qc(
    mask: np.ndarray, confidence: np.ndarray, used_canny_fallback: bool = False
) -> dict[str, float]:
    """Compute interpretable mask confidence and quality-control indicators."""
    binary = (mask > 0).astype(np.uint8)
    foreground = binary.astype(bool)
    confidence_mean = float(np.mean(confidence[foreground])) if foreground.any() else 0.0
    component_count = max(0, cv2.connectedComponents(binary, connectivity=8)[0] - 1)
    touches_border = bool(
        np.any(binary[0]) or np.any(binary[-1]) or np.any(binary[:, 0]) or np.any(binary[:, -1])
    )
    foreground_fraction = float(np.mean(binary))
    qc_score = confidence_mean
    if foreground_fraction < 0.001 or foreground_fraction > 0.999:
        qc_score *= 0.25
    if touches_border:
        qc_score *= 0.8
    if used_canny_fallback:
        qc_score *= 0.5
    return {
        "confidence": confidence_mean,
        "qc_score": float(np.clip(qc_score, 0.0, 1.0)),
        "foreground_fraction": foreground_fraction,
        "component_count": float(component_count),
        "touches_border": float(touches_border),
        "canny_fallback": float(used_canny_fallback),
    }


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return max(contours, key=cv2.contourArea) if contours else None


def contour_metrics(
    mask: np.ndarray, pixel_size: float = 1.0
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Extract every exterior contour and measure its physical arc length."""
    contours, _ = cv2.findContours(
        (np.asarray(mask) > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    lengths = np.asarray(
        [cv2.arcLength(contour, True) * pixel_size for contour in contours],
        dtype=float,
    )
    return contours, {
        "contour_count": float(len(contours)),
        "total_contour_length": float(np.sum(lengths)) if len(lengths) else 0.0,
        "mean_contour_length": float(np.mean(lengths)) if len(lengths) else 0.0,
        "max_contour_length": float(np.max(lengths)) if len(lengths) else 0.0,
    }


def pack_contours(contours: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pack variable-length contour coordinates into safe numeric NPZ arrays."""
    lengths = np.asarray([len(contour) for contour in contours], dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    if contours:
        points = np.concatenate([contour[:, 0, :] for contour in contours]).astype(np.int32)
    else:
        points = np.empty((0, 2), dtype=np.int32)
    return points, offsets


def contour_curvature(
    mask: np.ndarray, pixel_size: float = 1.0
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute signed d(theta)/ds curvature along the largest closed contour."""
    contour = largest_contour(mask)
    curvature_map = np.zeros(mask.shape, dtype=np.float32)
    if contour is None or len(contour) < 5:
        return curvature_map, {
            "mean_signed_curvature": np.nan,
            "mean_absolute_curvature": np.nan,
            "max_absolute_curvature": np.nan,
        }
    points = contour[:, 0, :].astype(np.float64) * pixel_size
    # Periodic central differences preserve the sign of d(theta)/ds.
    first = (np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)) / 2.0
    second = np.roll(points, -1, axis=0) - 2.0 * points + np.roll(points, 1, axis=0)
    denominator = np.power(first[:, 0] ** 2 + first[:, 1] ** 2, 1.5)
    numerator = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    coordinates = contour[:, 0, :]
    curvature_map[coordinates[:, 1], coordinates[:, 0]] = values.astype(np.float32)
    return curvature_map, {
        "mean_signed_curvature": float(np.mean(values)),
        "mean_absolute_curvature": float(np.mean(np.abs(values))),
        "max_absolute_curvature": float(np.max(np.abs(values))),
    }


def shape_descriptors(mask: np.ndarray, pixel_size: float = 1.0) -> dict[str, float]:
    """Measure area, perimeter, circularity, elongation, and orientation."""
    contour = largest_contour(mask)
    if contour is None:
        return {key: np.nan for key in ("area", "perimeter", "circularity", "elongation", "angle")}
    area = float(cv2.contourArea(contour) * pixel_size**2)
    perimeter = float(cv2.arcLength(contour, True) * pixel_size)
    circularity = float(4.0 * np.pi * area / perimeter**2) if perimeter > 0 else np.nan
    if len(contour) >= 5:
        (_, _), axes, angle = cv2.fitEllipse(contour)
        major, minor = max(axes), min(axes)
    else:
        (_, _), axes, angle = cv2.minAreaRect(contour)
        major, minor = max(axes), min(axes)
    elongation = float(major / minor) if minor > 0 else np.nan
    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "elongation": elongation,
        "angle": float(angle),
    }


def normalized_shape_change(records: list[dict[str, float]]) -> dict[str, np.ndarray]:
    """Return normalized delta time series relative to the first valid frame."""
    output: dict[str, np.ndarray] = {}
    for key in ("area", "perimeter", "circularity", "elongation", "angle"):
        values = np.asarray([record[key] for record in records], dtype=float)
        finite = np.flatnonzero(np.isfinite(values))
        if not len(finite):
            output[key] = np.full_like(values, np.nan)
            continue
        baseline = values[finite[0]]
        if key == "angle":
            delta = (values - baseline + 90.0) % 180.0 - 90.0
            output[key] = delta / 180.0
        else:
            scale = abs(baseline)
            output[key] = (values - baseline) / scale if scale > 1e-12 else values - baseline
    return output


def displacement_field(
    previous: np.ndarray,
    current: np.ndarray,
    method: str = "flow",
    window_size: int = 32,
    pixel_size: float = 1.0,
) -> np.ndarray:
    """Estimate dense displacement u(x,t) by flow or rigid registration."""
    before, after = as_uint8(previous), as_uint8(current)
    if before.shape != after.shape:
        raise ValueError("Deformation frames must have the same shape")
    if method == "registration":
        shift, _ = cv2.phaseCorrelate(before.astype(np.float32), after.astype(np.float32))
        field = np.empty((*before.shape, 2), dtype=np.float32)
        field[..., 0], field[..., 1] = shift
    elif method == "flow":
        field = cv2.calcOpticalFlowFarneback(
            before, after, None, 0.5, 3, window_size, 3, 5, 1.2, 0
        )
    else:
        raise ValueError("Deformation method must be flow or registration")
    return field.astype(np.float32) * pixel_size


def curl_map(displacement: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """Compute signed two-dimensional curl dv/dx - du/dy."""
    field = np.asarray(displacement, dtype=float)
    if field.ndim != 3 or field.shape[2] != 2:
        raise ValueError("Displacement must have shape (height, width, 2)")
    du_dy, _ = np.gradient(field[..., 0], spacing, spacing)
    _, dv_dx = np.gradient(field[..., 1], spacing, spacing)
    return (dv_dx - du_dy).astype(np.float32)


def strain_tensor(
    displacement: np.ndarray, spacing: float = 1.0, finite: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute small-strain or Green-Lagrange finite-strain tensor maps."""
    field = np.asarray(displacement, dtype=float)
    if field.ndim != 3 or field.shape[2] != 2:
        raise ValueError("Displacement must have shape (height, width, 2)")
    du_dy, du_dx = np.gradient(field[..., 0], spacing, spacing)
    dv_dy, dv_dx = np.gradient(field[..., 1], spacing, spacing)
    if finite:
        exx = du_dx + 0.5 * (du_dx**2 + dv_dx**2)
        eyy = dv_dy + 0.5 * (du_dy**2 + dv_dy**2)
        exy = 0.5 * (du_dy + dv_dx + du_dx * du_dy + dv_dx * dv_dy)
    else:
        exx = du_dx
        eyy = dv_dy
        exy = 0.5 * (du_dy + dv_dx)
    return exx.astype(np.float32), eyy.astype(np.float32), exy.astype(np.float32)


_NEIGHBORS = (
    (-1, -1, np.sqrt(2.0)), (-1, 0, 1.0), (-1, 1, np.sqrt(2.0)),
    (0, -1, 1.0), (0, 1, 1.0),
    (1, -1, np.sqrt(2.0)), (1, 0, 1.0), (1, 1, np.sqrt(2.0)),
)


def _skeleton_graph(skeleton: np.ndarray) -> tuple[dict[tuple[int, int], list], dict]:
    pixels = {tuple(point) for point in np.argwhere(skeleton)}
    graph: dict[tuple[int, int], list[tuple[tuple[int, int], float]]] = {}
    degree = {}
    for y, x in pixels:
        neighbors = []
        for dy, dx, weight in _NEIGHBORS:
            candidate = (y + dy, x + dx)
            if candidate in pixels:
                neighbors.append((candidate, weight))
        graph[(y, x)] = neighbors
        degree[(y, x)] = len(neighbors)
    return graph, degree


def _dijkstra_farthest(graph: dict, start: tuple[int, int]) -> tuple[tuple[int, int], float]:
    distances = {start: 0.0}
    queue = [(0.0, start)]
    farthest = start
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if distance > distances[farthest]:
            farthest = node
        for neighbor, weight in graph[node]:
            candidate = distance + weight
            if candidate < distances.get(neighbor, np.inf):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return farthest, distances[farthest]


def crack_metrics(mask: np.ndarray, pixel_size: float = 1.0) -> tuple[np.ndarray, dict]:
    """Measure skeleton length and local crack width from a binary crack mask.

    Width is twice the Euclidean distance from each skeleton pixel to the crack
    boundary. This is the local diameter of the maximal inscribed disk.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    skeleton = skeletonize(binary > 0)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    width_map = np.zeros(binary.shape, dtype=np.float32)
    width_map[skeleton] = 2.0 * distance[skeleton] * pixel_size
    widths = width_map[skeleton]
    graph, degree = _skeleton_graph(skeleton)
    if not graph:
        return skeleton.astype(np.uint8), {
            "length": 0.0,
            "longest_path": 0.0,
            "branch_count": 0,
            "tip_count": 0,
            "tips": [],
            "mean_width": 0.0,
            "max_width": 0.0,
            "width_map": width_map,
        }
    total = sum(weight for edges in graph.values() for _, weight in edges) / 2.0
    tips = [node for node, value in degree.items() if value == 1]
    branch_mask = np.zeros_like(skeleton, dtype=np.uint8)
    for node, value in degree.items():
        if value > 2:
            branch_mask[node] = 1
    branch_count = max(0, cv2.connectedComponents(branch_mask, connectivity=8)[0] - 1)
    candidates = tips or [next(iter(graph))]
    longest = 0.0
    # Exact endpoint-to-endpoint graph diameter for typical sparse crack masks.
    for tip in candidates:
        _, distance = _dijkstra_farthest(graph, tip)
        longest = max(longest, distance)
    tips_xy = [(int(x), int(y)) for y, x in tips]
    return skeleton.astype(np.uint8), {
        "length": float(total * pixel_size),
        "longest_path": float(longest * pixel_size),
        "branch_count": int(branch_count),
        "tip_count": len(tips_xy),
        "tips": tips_xy,
        "mean_width": float(np.mean(widths)) if len(widths) else 0.0,
        "max_width": float(np.max(widths)) if len(widths) else 0.0,
        "width_map": width_map,
    }


def track_crack_tips(
    previous_tips: list[tuple[int, int]],
    current_tips: list[tuple[int, int]],
    pixel_size: float = 1.0,
) -> dict[str, float]:
    """Track tips by nearest neighbor and summarize their displacement."""
    if not previous_tips or not current_tips:
        return {"mean_tip_displacement": np.nan, "max_tip_displacement": np.nan}
    previous = np.asarray(previous_tips, dtype=float)
    current = np.asarray(current_tips, dtype=float)
    distances = np.linalg.norm(previous[:, None, :] - current[None, :, :], axis=2)
    nearest = np.min(distances, axis=1) * pixel_size
    return {
        "mean_tip_displacement": float(np.mean(nearest)),
        "max_tip_displacement": float(np.max(nearest)),
    }
