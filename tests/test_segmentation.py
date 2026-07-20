import csv

import cv2
import numpy as np

from analysis.segmentation import analyze_segmentation, segment_frame
from core import (
    BarcodeConfig,
    ChannelResults,
    SegmentationConfig,
    SegmentationResults,
    WriterConfig,
)
from utils.reader import read_csv_to_channel_results
from utils.writer import results_to_csv


def synthetic_frame(size=128, value=4000):
    frame = np.zeros((size, size), dtype=np.uint16)
    cv2.rectangle(frame, (25, 20), (100, 105), value, thickness=-1)
    return frame


def test_segment_frame_detects_and_converts_uint16_data():
    edges, mask, measurements = segment_frame(
        synthetic_frame(), SegmentationConfig(minimum_segment_area=500)
    )

    assert edges.dtype == np.uint8
    assert mask.dtype == np.uint8
    assert measurements["segment_count"] == 1
    assert 0 < measurements["edge_density"] < measurements["segmented_area"] < 1


def test_video_analysis_writes_reduced_data(tmp_path):
    video = np.stack([synthetic_frame(value=2000 + index * 100) for index in range(6)])
    figure, results = analyze_segmentation(
        video,
        str(tmp_path),
        SegmentationConfig(frame_step=2, minimum_segment_area=500),
        WriterConfig(save_rds=True, save_visualizations=False),
    )

    assert figure is None
    assert results.mean_segment_count == 1
    assert results.mean_segmented_area > 0
    rds = (tmp_path / "SegmentationData.csv").read_text()
    assert "Edge Map" in rds
    assert "Segmentation Mask" in rds


def test_results_csv_round_trip_includes_segmentation(tmp_path):
    path = tmp_path / "results.csv"
    original = ChannelResults(
        filepath="sample.tif",
        channel=0,
        segmentation=SegmentationResults(
            mean_edge_density=0.1,
            max_edge_density=0.2,
            edge_density_change=1.1,
            mean_segmented_area=0.3,
            mean_segment_count=4.0,
        ),
    )
    results_to_csv([original], str(path))
    restored = read_csv_to_channel_results(str(path))[0]

    assert restored.segmentation.get_data() == original.segmentation.get_data()
    with path.open(newline="") as handle:
        headers = next(csv.reader(handle))
    assert headers[-5:] == SegmentationResults.get_headers()


def test_segmentation_config_yaml_round_trip(tmp_path):
    path = tmp_path / "settings.yaml"
    config = BarcodeConfig()
    config.modules.edge_segmentation = True
    config.segmentation_parameters.canny_low = 35
    config.save_to_yaml(str(path))

    restored = BarcodeConfig.load_from_yaml(str(path))
    assert restored.modules.edge_segmentation is True
    assert restored.segmentation_parameters.canny_low == 35
