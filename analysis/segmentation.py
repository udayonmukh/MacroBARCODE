"""Canny edge detection and closed-contour segmentation for microscopy videos."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from core import SegmentationConfig, SegmentationResults, WriterConfig
from utils import find_analysis_frames, vprint
from utils.setup import setup_csv_writer


def _as_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert integer or floating microscopy data to OpenCV-compatible uint8."""
    data = np.asarray(frame)
    if data.ndim != 2:
        raise ValueError("Segmentation expects a two-dimensional channel frame")
    finite = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if finite.dtype == np.uint8:
        return finite
    low, high = np.percentile(finite, (0.5, 99.5))
    if high <= low:
        return np.zeros(finite.shape, dtype=np.uint8)
    scaled = np.clip((finite - low) * (255.0 / (high - low)), 0, 255)
    return scaled.astype(np.uint8)


def _validate_config(config: SegmentationConfig) -> None:
    if not 0 <= config.canny_low < config.canny_high:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high")
    if config.blur_kernel < 1 or config.blur_kernel % 2 == 0:
        raise ValueError("Segmentation blur kernel must be a positive odd number")
    if config.close_kernel < 1:
        raise ValueError("Segmentation closing kernel must be positive")
    if config.close_iterations < 0 or config.minimum_segment_area < 0:
        raise ValueError("Segmentation iterations and minimum area cannot be negative")


def detect_edges(frame: np.ndarray, config: SegmentationConfig) -> np.ndarray:
    """Normalize, denoise, and apply Canny edge detection to one frame."""
    _validate_config(config)
    normalized = _as_uint8(frame)
    blurred = cv2.GaussianBlur(
        normalized, (config.blur_kernel, config.blur_kernel), sigmaX=0
    )
    return cv2.Canny(blurred, config.canny_low, config.canny_high)


def segment_frame(
    frame: np.ndarray, config: SegmentationConfig
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Convert edges into a filled binary mask and frame-level measurements."""
    edges = detect_edges(frame, config)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.close_kernel, config.close_kernel)
    )
    closed = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, kernel, iterations=config.close_iterations
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= config.minimum_segment_area
    ]
    mask = np.zeros_like(edges)
    if contours:
        cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
    pixel_count = mask.size
    measurements = {
        "edge_density": float(np.count_nonzero(edges) / pixel_count),
        "segmented_area": float(np.count_nonzero(mask) / pixel_count),
        "segment_count": len(contours),
    }
    return edges, mask, measurements


def analyze_segmentation(
    video: np.ndarray,
    name: str,
    config: SegmentationConfig,
    out_config: WriterConfig,
) -> Tuple[Optional[plt.Figure], SegmentationResults]:
    """Analyze selected frames and convert edge data into BARCODE metrics."""
    vprint("Beginning Edge Segmentation Analysis")
    _validate_config(config)
    frame_indices, _ = find_analysis_frames(video, config.frame_step)
    if len(frame_indices) == 0:
        return None, SegmentationResults()

    csvwriter = csvfile = None
    if out_config.save_rds:
        from visualization import write_segmentation_rds

        csvwriter, csvfile = setup_csv_writer(os.path.join(name, "SegmentationData.csv"))

    edge_density = []
    segmented_area = []
    segment_count = []
    save_spots = {int(frame_indices[0]), int(frame_indices[len(frame_indices) // 2]), int(frame_indices[-1])}

    for frame_idx in frame_indices:
        edges, mask, values = segment_frame(video[frame_idx], config)
        edge_density.append(values["edge_density"])
        segmented_area.append(values["segmented_area"])
        segment_count.append(values["segment_count"])

        if out_config.save_rds:
            write_segmentation_rds(
                csvwriter,
                (edges > 0).astype(np.uint8),
                (mask > 0).astype(np.uint8),
                int(frame_idx),
                [values["edge_density"], values["segmented_area"], values["segment_count"]],
            )
        if out_config.save_visualizations and int(frame_idx) in save_spots:
            from visualization import save_segmentation_visualization

            save_segmentation_visualization(video[frame_idx], edges, mask, int(frame_idx), name)

    if csvfile:
        csvfile.close()

    edge_values = np.asarray(edge_density, dtype=float)
    eval_count = max(1, int(np.ceil(len(edge_values) * config.percentage_frames_evaluated)))
    initial = float(np.mean(edge_values[:eval_count]))
    final = float(np.mean(edge_values[-eval_count:]))
    edge_change = final / initial if initial > 0 else np.nan

    fig = None
    if out_config.save_visualizations:
        from visualization import save_segmentation_plots

        fig = save_segmentation_plots(
            np.asarray(frame_indices), edge_values, np.asarray(segmented_area)
        )

    return fig, SegmentationResults(
        mean_edge_density=float(np.mean(edge_values)),
        max_edge_density=float(np.max(edge_values)),
        edge_density_change=edge_change,
        mean_segmented_area=float(np.mean(segmented_area)),
        mean_segment_count=float(np.mean(segment_count)),
    )
