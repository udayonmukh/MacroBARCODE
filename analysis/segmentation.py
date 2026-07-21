"""Mask-first segmentation and BARCODE mechanics orchestration."""

from __future__ import annotations

import csv
import os
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from analysis.mechanics import (
    as_uint8,
    boundary_from_mask,
    boundary_overlay,
    contour_curvature,
    contour_metrics,
    crack_metrics,
    curl_map,
    displacement_field,
    load_model_plugin,
    normalized_shape_change,
    pack_contours,
    segment_mask,
    segmentation_qc,
    shape_descriptors,
    strain_tensor,
    track_crack_tips,
    validate_config,
)
from core import (
    MechanicsResults,
    ReaderConfig,
    SegmentationConfig,
    SegmentationResults,
    WriterConfig,
)
from utils import find_analysis_frames, vprint
from utils.setup import setup_csv_writer


def detect_edges(frame: np.ndarray, config: SegmentationConfig) -> np.ndarray:
    """Return the mask contour, using Canny only if segmentation is empty."""
    mask, _ = segment_mask(frame, config)
    boundary, _ = boundary_from_mask(
        mask, frame, config.canny_low, config.canny_high
    )
    return boundary


def segment_frame(
    frame: np.ndarray, config: SegmentationConfig
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Segment one frame and return boundary, mask, and QC-safe measurements."""
    mask, confidence = segment_mask(frame, config)
    boundary, fallback = boundary_from_mask(
        mask, frame, config.canny_low, config.canny_high
    )
    qc = segmentation_qc(mask, confidence, fallback)
    measurements = {
        "edge_density": float(np.count_nonzero(boundary) / boundary.size),
        "segmented_area": float(np.count_nonzero(mask) / mask.size),
        "segment_count": int(qc["component_count"]),
        **qc,
    }
    return boundary, mask, measurements


def _last(series: np.ndarray) -> float:
    finite = series[np.isfinite(series)]
    return float(finite[-1]) if len(finite) else np.nan


def _mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.nanmean(array)) if np.isfinite(array).any() else np.nan


def _max(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.nanmax(array)) if np.isfinite(array).any() else np.nan


def analyze_segmentation(
    video: np.ndarray,
    name: str,
    config: SegmentationConfig,
    reader_config: ReaderConfig,
    out_config: WriterConfig,
) -> Tuple[Optional[plt.Figure], SegmentationResults, MechanicsResults]:
    """Compute segmentation and mechanics metrics plus auditable evidence maps."""
    vprint("Beginning Segmentation and Mechanics Analysis")
    validate_config(config)
    frame_indices, _ = find_analysis_frames(video, config.frame_step)
    if len(frame_indices) == 0:
        return None, SegmentationResults(), MechanicsResults()

    plugin = load_model_plugin(config.model_plugin) if config.method == "model" else None
    pixel_size = float(reader_config.um_pixel_ratio)
    evidence_dir = os.path.join(name, "Mechanics Evidence")
    if out_config.save_rds:
        os.makedirs(evidence_dir, exist_ok=True)

    rds_writer = rds_file = None
    series_file = None
    series_writer = None
    if out_config.save_rds:
        from visualization import write_segmentation_rds

        rds_writer, rds_file = setup_csv_writer(os.path.join(name, "SegmentationData.csv"))
        series_file = open(
            os.path.join(name, "MechanicsTimeSeries.csv"), "w", newline="", encoding="utf-8"
        )
        series_writer = csv.DictWriter(
            series_file,
            fieldnames=[
                "frame", "confidence", "qc_score", "foreground_fraction",
                "component_count", "canny_fallback", "area", "perimeter",
                "circularity", "elongation", "angle", "mean_signed_curvature",
                "mean_absolute_curvature", "max_absolute_curvature", "crack_length",
                "crack_longest_path", "crack_branches", "crack_tips",
                "mean_crack_width", "max_crack_width", "contour_count",
                "mean_contour_length", "total_contour_length", "max_contour_length",
            ],
        )
        series_writer.writeheader()

    edge_density: list[float] = []
    segmented_area: list[float] = []
    segment_count: list[float] = []
    confidence_values: list[float] = []
    qc_values: list[float] = []
    curvature_mean: list[float] = []
    curvature_max: list[float] = []
    shapes: list[dict[str, float]] = []
    cracks: list[dict] = []
    contour_records: list[dict[str, float]] = []
    masks: list[np.ndarray] = []
    tips: list[list[tuple[int, int]]] = []
    save_spots = {
        int(frame_indices[0]),
        int(frame_indices[len(frame_indices) // 2]),
        int(frame_indices[-1]),
    }

    for frame_idx in frame_indices:
        frame = video[int(frame_idx)]
        mask, confidence = segment_mask(frame, config, plugin)
        boundary, fallback = boundary_from_mask(
            mask, frame, config.canny_low, config.canny_high
        )
        qc = segmentation_qc(mask, confidence, fallback)
        curvature, curvature_stats = contour_curvature(mask, pixel_size)
        contours, contour_stats = contour_metrics(mask, pixel_size)
        contour_points, contour_offsets = pack_contours(contours)
        shape = shape_descriptors(mask, pixel_size)
        crack_mask = cv2.bitwise_not(mask) if config.crack_invert_mask else mask
        skeleton, crack = crack_metrics(crack_mask, pixel_size)
        overlay = boundary_overlay(frame, boundary)

        masks.append(mask)
        shapes.append(shape)
        cracks.append(crack)
        contour_records.append(contour_stats)
        tips.append(crack["tips"])
        edge_density.append(float(np.count_nonzero(boundary) / boundary.size))
        segmented_area.append(float(np.count_nonzero(mask) / mask.size))
        segment_count.append(qc["component_count"])
        confidence_values.append(qc["confidence"])
        qc_values.append(qc["qc_score"])
        curvature_mean.append(curvature_stats["mean_absolute_curvature"])
        curvature_max.append(curvature_stats["max_absolute_curvature"])

        if rds_writer:
            from visualization import write_segmentation_rds

            write_segmentation_rds(
                rds_writer,
                (boundary > 0).astype(np.uint8),
                (mask > 0).astype(np.uint8),
                int(frame_idx),
                [edge_density[-1], segmented_area[-1], segment_count[-1]],
            )
            np.savez_compressed(
                os.path.join(evidence_dir, f"frame_{int(frame_idx):05d}.npz"),
                mask=(mask > 0).astype(np.uint8),
                confidence=confidence.astype(np.float32),
                boundary=(boundary > 0).astype(np.uint8),
                curvature=curvature,
                crack_skeleton=skeleton,
                crack_width=crack["width_map"],
                contour_points=contour_points,
                contour_offsets=contour_offsets,
            )
            series_writer.writerow(
                {
                    "frame": int(frame_idx),
                    **{key: qc[key] for key in (
                        "confidence", "qc_score", "foreground_fraction", "component_count", "canny_fallback"
                    )},
                    **shape,
                    **curvature_stats,
                    "crack_length": crack["length"],
                    "crack_longest_path": crack["longest_path"],
                    "crack_branches": crack["branch_count"],
                    "crack_tips": crack["tip_count"],
                    "mean_crack_width": crack["mean_width"],
                    "max_crack_width": crack["max_width"],
                    **contour_stats,
                }
            )
        if out_config.save_visualizations and int(frame_idx) in save_spots:
            from visualization import save_segmentation_visualization

            save_segmentation_visualization(frame, boundary, mask, int(frame_idx), name)
            cv2.imwrite(
                os.path.join(name, f"Boundary Overlay Frame {int(frame_idx)}.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

    displacement_means: list[float] = []
    displacement_maxima: list[float] = []
    curl_means: list[float] = []
    strain_xx: list[float] = []
    strain_yy: list[float] = []
    strain_xy: list[float] = []
    tip_displacements: list[float] = []

    for pair_index in range(len(frame_indices) - 1):
        start, stop = int(frame_indices[pair_index]), int(frame_indices[pair_index + 1])
        displacement = displacement_field(
            video[start],
            video[stop],
            method=config.deformation_method,
            window_size=config.deformation_window,
            pixel_size=pixel_size,
        )
        curl = curl_map(displacement, spacing=pixel_size)
        exx, eyy, exy = strain_tensor(
            displacement,
            spacing=pixel_size,
            finite=config.strain_type == "finite",
        )
        magnitude = np.linalg.norm(displacement, axis=2)
        tip_track = track_crack_tips(tips[pair_index], tips[pair_index + 1], pixel_size)
        displacement_means.append(float(np.mean(magnitude)))
        displacement_maxima.append(float(np.max(magnitude)))
        curl_means.append(float(np.mean(curl)))
        strain_xx.append(float(np.mean(exx)))
        strain_yy.append(float(np.mean(eyy)))
        strain_xy.append(float(np.mean(exy)))
        tip_displacements.append(tip_track["mean_tip_displacement"])
        if out_config.save_rds:
            np.savez_compressed(
                os.path.join(evidence_dir, f"pair_{start:05d}_{stop:05d}.npz"),
                displacement=displacement,
                curl=curl,
                strain_xx=exx,
                strain_yy=eyy,
                strain_xy=exy,
            )

    for handle in (rds_file, series_file):
        if handle:
            handle.close()

    edge_values = np.asarray(edge_density)
    eval_count = max(1, int(np.ceil(len(edge_values) * config.percentage_frames_evaluated)))
    initial_edge = float(np.mean(edge_values[:eval_count]))
    final_edge = float(np.mean(edge_values[-eval_count:]))
    edge_change = final_edge / initial_edge if initial_edge > 0 else np.nan
    changes = normalized_shape_change(shapes)
    crack_lengths = np.asarray([item["length"] for item in cracks], dtype=float)
    crack_change = (
        (crack_lengths[-1] - crack_lengths[0]) / abs(crack_lengths[0])
        if len(crack_lengths) and crack_lengths[0] > 0
        else np.nan
    )

    fig = None
    if out_config.save_visualizations:
        from visualization import save_segmentation_plots

        fig = save_segmentation_plots(
            np.asarray(frame_indices), edge_values, np.asarray(segmented_area)
        )

    segmentation_results = SegmentationResults(
        mean_edge_density=float(np.mean(edge_values)),
        max_edge_density=float(np.max(edge_values)),
        edge_density_change=edge_change,
        mean_segmented_area=float(np.mean(segmented_area)),
        mean_segment_count=float(np.mean(segment_count)),
    )
    mechanics_results = MechanicsResults(
        segmentation_confidence=_mean(confidence_values),
        segmentation_qc=_mean(qc_values),
        mean_absolute_curvature=_mean(curvature_mean),
        max_absolute_curvature=_max(curvature_max),
        area_change=_last(changes["area"]),
        perimeter_change=_last(changes["perimeter"]),
        circularity_change=_last(changes["circularity"]),
        elongation_change=_last(changes["elongation"]),
        angle_change=_last(changes["angle"]),
        mean_displacement=_mean(displacement_means),
        max_displacement=_max(displacement_maxima),
        mean_curl=_mean(curl_means),
        mean_strain_xx=_mean(strain_xx),
        mean_strain_yy=_mean(strain_yy),
        mean_strain_xy=_mean(strain_xy),
        max_crack_length=_max(crack_lengths.tolist()),
        crack_length_change=float(crack_change),
        max_crack_branches=_max([item["branch_count"] for item in cracks]),
        mean_crack_tip_displacement=_mean(tip_displacements),
        mean_contour_length=_mean([item["mean_contour_length"] for item in contour_records]),
        total_contour_length=_max([item["total_contour_length"] for item in contour_records]),
        mean_crack_width=_mean([item["mean_width"] for item in cracks]),
        max_crack_width=_max([item["max_width"] for item in cracks]),
    )
    return fig, segmentation_results, mechanics_results
