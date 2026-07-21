import os, functools, builtins
from itertools import pairwise
import nd2, av
import imageio.v3 as iio
import numpy as np
from utils import vprint
from core import (
    BarcodeConfig,
    InputConfig,
    ChannelResults,
    BinarizationResults,
    IntensityResults,
    FlowResults,
    SegmentationResults,
    MechanicsResults,
)

def check_first_frame_dim(file):
    min_intensity = np.min(file[0])
    mean_intensity = np.mean(file[0])
    return 2 * np.exp(-1) * mean_intensity <= min_intensity

def read_file(filepath, count_list, config: BarcodeConfig = None, in_config: InputConfig = None, accept_dim: bool = False, allow_large_files = True):
    print = functools.partial(builtins.print, flush=True)
    
    if count_list[1] != 1:    
        print(f'File {count_list[0]} of {count_list[1]}')
        print(filepath)
        count_list[0] += 1

    file_size = os.path.getsize(filepath)
    file_size_gb = file_size / (1024 ** 3)
    if file_size_gb > 5 and not allow_large_files:
        print("File size is too large -- this program does not process files larger than 5 GB.")
        return None
    if filepath.endswith(('.avi', '.mp4')):
        frames = []
        container = av.open(filepath)
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='gray'))
        file = np.array(frames)
        file = np.reshape(file, (file.shape + (1,))) if len(file.shape) == 3 else file
    if filepath.endswith(('.tif', '.tiff')):
        file = iio.imread(filepath)
        file = np.reshape(file, (file.shape + (1,))) if len(file.shape) == 3 else file
        if file.shape[3] != min(file.shape):
            file = np.swapaxes(np.swapaxes(file, 1, 2), 2, 3)
    elif filepath.endswith('.nd2'):
        with nd2.ND2File(filepath) as ndfile:
            if len(ndfile.sizes) >= 5:
                count_list[0] += 1
                raise TypeError("Incorrect file dimensions: file must be time series data with 1+ channels (4 dimensions total)")
            if "Z" in ndfile.sizes:
                count_list[0] += 1
                raise TypeError('Z-stack identified, skipping to next file...')
            if 'T' not in ndfile.sizes or len(ndfile.shape) <= 2 or ndfile.sizes['T'] <= 5:
                count_list[0] += 1
                raise TypeError('Too few frames, unable to capture dynamics, skipping to next file...')
            if ndfile == None:
                raise TypeError('Unable to read file, skipping to next file...')
            file = ndfile.asarray()
            if 'C' not in ndfile.sizes:
                file = np.expand_dims(file, axis=1)
            file = np.swapaxes(np.swapaxes(file, 1, 2), 2, 3)
            try:
                times = ndfile.events(orient="list")["Time [s]"]
                frame_interval = np.array([y - x for x, y in pairwise(times)]).mean()
                micron_pix_ratio = ndfile.voxel_size()[0]
                config.reader.exposure_time = float(frame_interval / in_config.time)
                config.reader.um_pixel_ratio = micron_pix_ratio / in_config.length
                vprint(f"Extracted ND2 metadata: frame_interval={frame_interval:.4f}s, micron_pixel_ratio={micron_pix_ratio:.2f}")
            except Exception as e:
                config.reader.exposure_time = 1
                config.reader.um_pixel_ratio = 1
                vprint(f"Warning: Could not extract ND2 metadata: {e}")
    if (file == 0).all():
        print('Empty file: can not process, skipping to next file...')
        return None
    
    if accept_dim == False and check_first_frame_dim(file) == True:
        print(filepath + 'is too dim, skipping to next file...')
        return None
    else:
        return file
    
def read_csv_to_channel_results(filepath: str) -> list[ChannelResults]:
    """Read current or legacy BARCODE result CSVs by metric name."""
    import csv

    def number(row: dict, name: str) -> float:
        value = row.get(name, "")
        if value is None or value == "" or value.lower() == "nan":
            return np.nan
        try:
            return float(value)
        except ValueError:
            return np.nan

    results = []
    with open(filepath, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        headers = reader.fieldnames or []
        if headers[:3] != ["File", "Channel", "Flags"]:
            raise AssertionError(f"CSV headers {headers} do not match BARCODE output")

        physical = "Maximum Island Area Quantity" in headers
        for row in reader:
            channel = number(row, "Channel")
            if np.isnan(channel):
                raise ValueError(f"Invalid channel in row: {row}")

            binarization = BinarizationResults(
                connectivity=number(row, "Connectivity"),
                max_island_percent_change=number(row, "Maximum Island Area Change"),
                max_void_percent_change=number(row, "Maximum Void Area Change"),
                island_anisotropy=number(row, "Mean Island Anisotropy"),
                mean_island_separation=number(row, "Mean Island Separation"),
                island_correlation_length=number(row, "Structural Correlation Length"),
            )
            if physical:
                binarization.max_island_size_quantity = number(row, "Maximum Island Area Quantity")
                binarization.max_void_size_quantity = number(row, "Maximum Void Area Quantity")
                binarization.island_size_initial_quantity = number(row, "Initial Maximum Island Area Quantity")
                binarization.island_size_initial2_quantity = number(row, "Initial 2nd Maximum Island Area Quantity")
                binarization.mean_island_size_quantity = number(row, "Mean Island Area Quantity")
                binarization.total_island_size_quantity = number(row, "Total Island Area Quantity")
            else:
                binarization.max_island_size = number(row, "Maximum Island Area")
                binarization.max_void_size = number(row, "Maximum Void Area")
                binarization.island_size_initial = number(row, "Initial Maximum Island Area")
                binarization.island_size_initial2 = number(row, "Initial 2nd Maximum Island Area")
                binarization.mean_island_size = number(row, "Mean Island Area")
                binarization.total_island_size = number(row, "Total Island Area")

            results.append(ChannelResults(
                filepath=row["File"],
                channel=int(channel),
                total_flags=row.get("Flags", "0"),
                binarization=binarization,
                intensity=IntensityResults(
                    max_kurtosis=number(row, "Maximum Kurtosis"),
                    max_median_skew=number(row, "Maximum Median Skewness"),
                    max_mode_skew=number(row, "Maximum Mode Skewness"),
                    kurtosis_diff=number(row, "Kurtosis Change"),
                    median_skew_diff=number(row, "Median Skewness Change"),
                    mode_skew_diff=number(row, "Mode Skewness Change"),
                ),
                flow=FlowResults(
                    mean_speed=number(row, "Speed"),
                    delta_speed=number(row, "Speed Change"),
                    mean_theta=number(row, "Mean Flow Direction"),
                    mean_sigma_theta=number(row, "Directional Spread"),
                    velocity_correlation_length=number(row, "Velocity Correlation Length"),
                    divergence=number(row, "Divergence"),
                    curl=number(row, "Curl"),
                ),
                segmentation=SegmentationResults(
                    mean_edge_density=number(row, "Mean Edge Density"),
                    max_edge_density=number(row, "Maximum Edge Density"),
                    edge_density_change=number(row, "Edge Density Change"),
                    mean_segmented_area=number(row, "Mean Segmented Area"),
                    mean_segment_count=number(row, "Mean Segment Count"),
                ),
                mechanics=MechanicsResults(
                    segmentation_confidence=number(row, "Segmentation Confidence"),
                    segmentation_qc=number(row, "Segmentation QC Score"),
                    mean_absolute_curvature=number(row, "Mean Absolute Curvature"),
                    max_absolute_curvature=number(row, "Maximum Absolute Curvature"),
                    area_change=number(row, "Shape Area Change"),
                    perimeter_change=number(row, "Shape Perimeter Change"),
                    circularity_change=number(row, "Shape Circularity Change"),
                    elongation_change=number(row, "Shape Elongation Change"),
                    angle_change=number(row, "Shape Angle Change"),
                    mean_displacement=number(row, "Mean Displacement"),
                    max_displacement=number(row, "Maximum Displacement"),
                    mean_curl=number(row, "Mechanics Curl"),
                    mean_strain_xx=number(row, "Mean Strain XX"),
                    mean_strain_yy=number(row, "Mean Strain YY"),
                    mean_strain_xy=number(row, "Mean Strain XY"),
                    max_crack_length=number(row, "Maximum Crack Length"),
                    crack_length_change=number(row, "Crack Length Change"),
                    max_crack_branches=number(row, "Maximum Crack Branch Count"),
                    mean_crack_tip_displacement=number(row, "Mean Crack Tip Displacement"),
                    mean_contour_length=number(row, "Mean Contour Length"),
                    total_contour_length=number(row, "Total Contour Length"),
                    mean_crack_width=number(row, "Mean Crack Width"),
                    max_crack_width=number(row, "Maximum Crack Width"),
                ),
            ))
    return results
