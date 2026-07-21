# Table of Contents
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Running BARCODE \& User Settings](#running-barcode--user-settings)
    - [Execution Settings](#execution-settings)
    - [Binarization Settings](#binarization-settings)
    - [Optical Flow Settings](#optical-flow-settings)
    - [Segmentation / Morphology Settings](#segmentation--morphology-settings)
    - [Intensity Distribution Settings](#intensity-distribution-settings)
    - [Barcode Generator + CSV Aggregator](#barcode-generator--csv-aggregator)
    - [Barcode Metric Comparison](#barcode-metric-comparison)
    - [Reduced Data Structure Visualization](#reduced-data-structure-visualization)
- [Outputs](#outputs)
  - [Metrics](#metrics)
    - [Binarization Metrics](#binarization-metrics)
    - [Optical Flow Field Metrics](#optical-flow-field-metrics)
    - [Segmentation / Morphology Metrics](#segmentation--morphology-metrics)
    - [Intensity Distribution Metrics](#intensity-distribution-metrics)
  - [Output Files](#output-files)

Before running BARCODE we recommend going through our detailed **[BARCODE 2.0 Tutorial](https://www.livingbam.org/barcode-2-tutorial)** which includes test data. It should take 10-15 seconds on a standard desktop computer to analyze the test data in the tutorial using the software.

# Installation
The BARCODE source code can be directly downloaded on any system running Python 3.12 or later from the [BARCODE-HTP Github repository](https://github.com/BARCODE-HTP/barcode). To install the required packages, you can use PIP to install them using the following command: ```pip install -r requirements.txt```. Keep in mind that this code was developed in Python 3.12 -- versions of Python prior to 3.12 may not be able to run this program from the source code. If running from source and an error indicates that the `av` package is missing, install it with ```pip install av```.  To run BARCODE, navigate to the source code directory in the terminal and run the command: ```python main.py``` or ```python3 main.py```.

# Usage
## Data Preparation
High-quality microscopy videos with minimal spurious signals should be used. If needed, videos should be cropped or trimmed prior to analysis. BARCODE accepts TIFF, ND2, AVI, and MP4 files. If files you wish to process are not in one of these formats, convert them to a supported format using ImageJ/FIJI or another video/image conversion tool.

## Running BARCODE & User Settings
Navigate to the source code directory in the terminal and run the command: ```python main.py``` or ```python3 main.py```. A window will appear with the graphical user interface (GUI). The user inputs are described below. When finished specifying the operational settings, click "Run" to begin the BARCODE program. 


Each branch also contains a live preview of the output, showing visualizations for Image Binarization, Optical Flow, Segmentation / Morphology, and Intensity Distribution. These previews update when parameters are modified, and include a slider for easy visualization throughout the video.


### Execution Settings
| Setting Name                      | Description                                                                                                                                                                                                                                           |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Select Data** | |
| Process File | Select a file or folder to run the BARCODE program on|
| Process Directory | Select a folder to run the BARCODE program on (can not be combined with the "Process File" option above) |
| **Select Channels** | |
| Choose Channel                 | Select a channel to run the program on (-1 for last channel, -2 for second to last channel, 0 for first channel, etc) |
| Parse All Channels | Analyze all channels for each video (can not be combined with the "Choose Channel" option) |
| **Specify Metadata** | |
| Micron to Pixel Ratio | Controls the ratio of microns to pixels in the image; if ND2 files are evaluated, this is taken from the metadata instead; used to adjust optical flow output units from pixels/flow field to microns/second |
| Exposure Time (seconds) | Controls the interval (in seconds) between frames; if ND2 files are evaluated, this is taken from the metadata instead; used to adjust optical flow output units from pixels/flow field to microns/second |
| Length Unit | Selects the physical length unit used for unit conversion and output labels, such as nm, Î¼m, or mm |
| Time Unit | Selects the time unit used for speed conversion and output labels, such as s, min, or hr |
| **Select Branches** |
| Image Binarization | Analyze all chosen videos with the Image Binarization branch | 
| Optical Flow | Analyze all chosen videos with the Optical Flow branch |
| Intensity Distribution | Analyze all chosen videos with the Intensity Distribution branch |
| Segmentation / Morphology | Analyze segmented material/object morphology, masked dynamics, mechanics, strain, and crack metrics |
| **Handling Dim Data** | |
| Scan Dim Files | Run the program on files that are dim (defined as videos where the mean pixel intensity in the first frame is less than $\frac{2}{e}$ times the minimum pixel intensity) -- files meeting this criteria are labeled in the BARCODE CSV file under the Flags section with a numerical label of 1 |
| Scan Dim Channels | Run the program on channels that are dim (defined in "Include Dim Files" setting) -- video channels meeting this criteria are labeled in the BARCODE CSV file under the Flags section (described in "Include Dim Files" setting) |
| **Output Settings** | 
| Verbose | Prints more details while running the program to output display, including modules run on videos, time to analyze files, etc. |
| Save Graphs | Saves branch visualizations as .png files for further analysis |
| Save Reduced Data Structures | Saves reduced data structures used to perform computation of metrics |
| Generate Dataset Barcode | Save a color "barcode" visualization of the entire dataset; useful for visualizing differences between videos |
| **Configuration Settings** |
| Configuration File | Select a Configuration YAML file; overwrite all settings selected by the user with settings from input YAML file |

### Binarization Settings
The Image Binarization branch takes frames from the original video and binarizes those frames. Following this, the binarized video is broken into connected components, with the growth of "voids" (connected components labelled as 0) and "islands" (connected components labelled as 1) measured. A live preview of the binarization is shown in the program to enable qualitative analysis of the effect of the binarization and the sensitivity to the binarization threshold.

| Setting Name | Description | Limits | Default Value |
| - | - | - | - |
| Binarization Threshold | Controls the threshold percentage of the mean which binarizes the image; offset parameter determines the binarization threshold for a given frame as $(1 + \text{offset}) * \overline{B(i)}$, where $\overline{B(i)}$ represents the mean pixel intensity for frame $i$ | (-1, 1) | 0.1 |
| Binning Ratio | Controls the extent of spatial downsampling performed on the image before binarization; averages windows of $p \times p$ pixels to reduce size of data | (1, 8) | 2 |
| Output Unit Conversion | Changes selected image binarization area metrics from field-of-view percentage units to physical area units based on the selected length unit | (On, Off) | Off |
| Frame Step | Controls the interval between binarized frames; affects speed of program, with larger intervals decreasing program runtime at potential loss of accuracy | (1, 100) | 10 |
| Fraction of Frames Evaluated | Used for determining frames for averaging in calculation of initial maximum island area and maximum island/void area change; not used for calculation of maximum island/void area; decreasing this results in fewer frames being used for these averages, at the cost of more sensitivity to noise | (0.01, 0.25) | 0.05 |

### Optical Flow Settings
The optical flow module takes frames from a video file and calculates the optical flow field using Farneback's Optical Flow algorithm.

| Setting Name | Description | Limits | Default Value |
| - | - | - | - |
| Frame Step | Controls the interval between frames with which the flow field is calculated; larger values are less prone to noise, but have less precision | (1, 100) | 10 |
| Optical Flow Window Size | Controls the window size used to compute the flow fields, described further in the [OpenCV documentation here](https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af) | (1, 1000) | 32 |
| Downsample | Controls the interval between pixels that the flow field is sampled at; larger values are less prone to noise but have less precision | (1, 1000) | 8 |
| Fraction of Frames Evaluated | Used for determining frames for averaging in calculation of speed change; not used for calculation of other optical flow metrics; decreasing this results in fewer frames being used for these averages, at the cost of more sensitivity to noise | (0.01, 0.25) | 0.05 |

### Segmentation / Morphology Settings
The Segmentation / Morphology branch segments material or object regions from TIFF, ND2, AVI, and MP4 inputs. It follows the selected-channel or all-channels setting used by the rest of BARCODE. The branch can generate masks, contours, connected components, object labels, trajectories, masked optical-flow metrics, mechanics metrics, strain metrics, and crack metrics.

Edge density is defined as:

```text
Edge Density = Total Boundary Length / Total Material Area
```

It measures how boundary-rich or fragmented the segmented material is. Higher edge density indicates more boundary per unit segmented material area.

Mechanics and crack CSV outputs are computed whenever the Segmentation / Morphology branch runs. The mechanics/crack visualization option only controls whether additional PNG and evidence visualizations are saved.

| Setting Name | Description | Default Value |
| - | - | - |
| Threshold Method | Segmentation method: Otsu, Mean, Percentile, Adaptive, or Canny | Otsu |
| Threshold Offset | Offset applied to global threshold methods | 0 |
| Invert Segmentation | Segment dark material/object regions instead of bright regions | Off |
| Adaptive Block Size / Adaptive C | Local neighborhood size and threshold offset for adaptive thresholding | 31 / 5 |
| Canny Lower / Upper Threshold | Edge thresholds used by the Canny method | 50 / 150 |
| Smoothing Sigma | Gaussian smoothing before thresholding or edge detection | 1 |
| Minimum Object Size | Remove segmented objects smaller than this area in pixels | 16 |
| Hole Filling Size | Fill holes smaller than this area in pixels | 16 |
| Frame Step | Analyze every Nth frame | 10 |
| Fraction of Frames Evaluated | Beginning/end fraction used for temporal summary metrics | 0.05 |
| Save Labeled Segmentation Video | Save a video showing segmented objects with labels | Off |
| Save Component Trajectories | Save object-level persistent label trajectories | Off |
| Save Masked Flow Metrics | Save detailed full-FOV versus mask-restricted optical-flow CSV outputs | Off |
| Save Masked Flow Visualizations | Save first, middle, and final masked-flow comparison images | Off |
| Save Mechanics / Crack Visualizations | Save mechanics/crack evidence PNGs; mechanics/crack CSVs are computed by default | Off |
| Treat Mask Background as Crack Network | Skeletonize the mask background instead of the foreground for crack measurements | Off |
| Deformation Method | Use dense optical flow or registration for deformation measurement | flow |
| Deformation Window | Window size used by the deformation calculation | 32 |
| Strain Tensor | Use small-strain or finite-strain tensor output | small |

The primary segmentation-derived metrics included in `Summary.csv` and BARCODE are:

- `Segmentation Edge Density Max`
- `Segmentation Edge Density Mean`
- `Segmentation Masked Mean Speed`
- `Segmentation Strain XX`
- `Segmentation Strain YY`
- `Segmentation Strain XY`
- `Segmentation Crack Length Change`
- `Segmentation Crack Length Maximum`
- `Segmentation Crack Width Maximum`

Masked mean speed measures optical-flow speed inside segmented masks while excluding background pixels. Strain XX, YY, and XY describe deformation components from mechanics analysis. Crack length and crack width metrics quantify crack network extent and opening.

#### Segmentation / Morphology CSV Outputs

**Segmentation / Morphology**

- `SegmentationMetrics.csv`: Per-frame global segmentation metrics, including material area, component count, total perimeter, edge density, boundary roughness, solidity, and boundary displacement.
- `SegmentationComponents.csv`: Per-frame properties for each segmented object, including area, centroid, bounding box, major/minor axis length, orientation, and solidity.
- `SegmentationMasks.csv`: Per-frame segmentation masks stored as 0/1 matrices.
- `SegmentationComponentTrajectories.csv`: Persistent label trajectories for segmented objects across frames.

**Masked Dynamic**

- `MaskedFlowMetrics.csv`: Comparison between full-FOV optical flow and mask-restricted optical flow, including masked mean speed, direction, and directional coherence.
- `MaskedFlowComponents.csv`: Optical-flow summary metrics computed within each segmented component.

**Mechanics / Crack**

- `MechanicsTimeSeries.csv`: Per-frame mechanics and crack metrics, such as crack length, crack width, contour curvature, and shape descriptors.
- `MechanicsPairMetrics.csv`: Frame-pair deformation metrics between consecutive analyzed frames, such as displacement, curl, and strain components XX, YY, and XY.

Important segmentation visual outputs include:

- `Segmentation Frame X Overlay.png`
- `Segmentation Frame X Mask.png`
- `Segmentation Frame X Labeled Components.png`
- `Segmentation First Middle Last Components.png`
- `Segmentation Edge Density.png`
- `Segmentation Boundary Displacement.png`
- `Segmentation Centroid Trajectory.png`
- `Segmentation Labeled Components.mp4`
- `Masked Flow Comparison Frame X to Y.png`
- `Mechanics Evidence Frame X.png`
- `Mechanics Pair Frame X to Y.png`

`Segmentation Boundary Displacement.png` is saved as a standalone diagnostic visualization when visualizations are enabled, but boundary displacement is not currently included in Summary Graphs or BARCODE metrics.

The settings tab provides a live three-panel preview of the original frame, mask boundary, and filled segmentation mask. Parameters update immediately, and the preview reports segmentation quality, shape, edge density, and crack measurements.

### Intensity Distribution Settings
The intensity distribution module takes frames from a video and creates an intensity distribution histogram. The kurtosis, median skewness, and mode skewness are then calculated from this distribution.

| Setting Name | Description | Limits | Default Value |
| - | - | - | - |
| Frame Step | Controls the interval between frames for which the intensity distributions are calculated; affects speed of program, with larger intervals decreasing program runtime | (1, 100) | 10 |
| Distribution Number of Bins | Controls the number of bins in histogram; increasing/decreasing the number of bins may result in binning artifacts that affect accuracy of intensity distribution | (100, 500) | 300 |
| Distribution Noise Threshold | Controls the minimum normalized probability in the intensity distribution; probabilities below this threshold are set to 0 and the distribution is renormalized; increasing/decreasing this will affect the sensitivity of the metrics to noise, which is particularly relevant for higher pixel intensity values | (0.00001, 0.01) | 0.0005 |
| Fraction of Frames Evaluated | Used for determining frames for averaging in calculation of kurtosis/median skewness/mode skewness change; not used for calculation of maximum kurtosis/median skewness/mode skewness metrics; decreasing this results in fewer frames being used for these averages, at the cost of more sensitivity to noise | (0.01, 0.25) | 0.05 |

### Barcode Generator + CSV Aggregator
| Setting Name |  Description |
| - | - |
| CSV File Locations | Select the CSV files representing the datasets you would like to combine |
| Aggregate Location | Select a location for the aggregate CSV file to be located |
| Generate Aggregate Barcode | Controls whether or not a colorized barcode is generated |
| Generate Barcodes for Comparison | For use if selecting multiple CSV files for comparison; generates barcodes for each input CSV file with consistent limits for direct comparison of metrics |
| Sort Parameter | Determines which metric is used to sort the colorized barcode; if left on Default, the barcode will be sorted by filename/order of files within the CSV files aggregated |
| Select Metrics to Visualize in BARCODE | Select metrics shown in the colorized barcode; if left unmodified, all metrics will be included in the output barcode |

### Barcode Metric Comparison
| Setting Name | Description |
| - | - |
| Select CSV File | Choose a CSV file to compare two BARCODE parameters |
| Metric 1 | Select a metric for comparison |
| Metric 2 | Select a second metric for comparison |
| CSV Comparison Location | Select a location for a CSV file containing the two metrics being compared to be saved to

### Reduced Data Structure Visualization
| Setting Name | Description |
| - | - |
| Select RDS File | Choose an RDS file from an analysis branch to preview |

# Outputs
## Metrics
Each module contributes testable summary metrics to the BARCODE analysis. They are described below:
### Binarization Metrics
The Binarization module uses a binarization threshold (defined [above](#binarization-settings)) to convert each selected frame from a given video from grayscale to 0's and 1's. This binarized image is then segmented into "islands" (a region comprising of only 1's) and "voids" (a region comprising of only 0's). The following metrics are computed with respect to these definitions.
| Metric | Description  |
| - | - |
| **Connectivity** | The percentage of frames that are defined as "connected" (there exists a single island spanning from the top to bottom of the frame, or from the left to right of the frame) |
| **Maximum Island Area** | Fractional area of the largest island in the video; calculated by averaging fractional area of the largest island in each frame over the frames with the top 10% largest islands |
| **Maximum Void Area** | Fractional area of the largest void; calculated in a similar manner to the ***Maximum Island Area*** metric |
| **Maximum Island Area Change** | The percentage growth/shrinkage of the largest island; calculated by averaging the island area over the first *X* percent and last *X* percent of frames and calculating the difference between these two averages |
| **Maximum Void Area Change** | The percentage growth/shrinkage of the largest void; calculated in a similar manner to the ***Maximum Island Area Change*** metric |
| **Initial Maximum Island Area** | Fractional area of the largest island in the first *X* percent of frames; used as a measurement of the heterogeneity of the island areas in the frame |
| **Initial 2nd Maximum Island Area** | Fractional area of the second largest island in the first *X* percent of frames; used in combination with ***Initial Maximum Island Area*** as a measurement of the heterogeneity of the connected components in the frame |
| **Mean Island Anisotropy** | The average anisotropy of all islands in a given frame, averaged over all frames; calculated using the quotient of the major and minor axis lengths |
| **Mean Island Area** | The average fractional area of all islands in a given frame, averaged over all frames |
| **Total Island Area** | The total fractional area of all islands in a given frame, averaged over all frames |
| **Mean Island Separation** | The average distance between all islands in a given frame, averaged over all frames; calculated by calculating the distance between each island centroid |
| **Structural Correlation Length** | The distance $r$ where the average correlation in pixel intensity between two pixels separated by distance $r$ drops below $\frac{1}{e}$, averaged over all frames; calculated using Fast Fourier Transforms to determine the correlation and then radially averaged to find $r$, before averaging for all frames |

### Optical Flow Field Metrics
The Optical Flow module computes a "flow field" for pairs of frames as selected by the user, with each flow field consisting of a $m/p$ x $n/p$ grid of velocity vectors, where $m$ x $n$ are the dimensions of a given frame, and $p$ is the downsampling factor (defined [above](#optical-flow-settings)). The following metrics are computed with respect to these definitions.
| Metric | Description |
| - | - |
| **Speed** | The average speed over all flow fields in the video; calculated by taking the magnitude of each velocity vector and averaging over the flow field, before averaging the output of each flow field. |
| **Speed Change** | The change in the average speed throughout the video; calculated in a similar manner to the ***Maximum Island Area Change*** metrics. |
| **Mean Flow Direction** | The average direction of the velocity vectors throughout the video; calculated by normalizing the velocity vectors to unit vectors, then taking the average of the $x$ and $y$ components for each vector separately over all flow fields, and then taking the two-argument arctan2 function to calculate an average flow direction. |
| **Directional Spread** | The average variance in the direction of the velocity vectors throughout the video; calculated by taking the unit velocity vectors described in the ***Mean Flow Direction***, then the calculating the length of the mean vector for each flow field. Using the definition of circular variance, the variance is then averaged over all frames. |
| **Velocity Correlation Length** | The distance $r$ where the average correlation in velocity between two pixels separated by distance $r$ drops below $0.5$, averaged over all flow fields; calculated using direct computation with the formula $\frac{\langle V(r) \cdot V(0)\rangle}{\langle \|V(0)\|^2\rangle}$ |
| **Divergence** | The average spatial divergence of the cumulative velocity field, used to describe local expansion or contraction in the flow field |
| **Curl** | The average curl of the normalized velocity field, used to describe local rotational motion in the flow field |

### Segmentation / Morphology Metrics
The Segmentation / Morphology branch contributes summary metrics that describe material boundary density, mask-restricted motion, deformation, and crack geometry.

| Metric | Description |
| - | - |
| **Segmentation Edge Density Max** | Maximum edge density across analyzed frames, where edge density is total boundary length divided by total material area |
| **Segmentation Edge Density Mean** | Mean edge density across analyzed frames |
| **Segmentation Masked Mean Speed** | Mean optical-flow speed inside segmented masks, excluding background pixels |
| **Segmentation Strain XX** | Mean x-direction strain from mechanics analysis |
| **Segmentation Strain YY** | Mean y-direction strain from mechanics analysis |
| **Segmentation Strain XY** | Mean shear strain from mechanics analysis |
| **Segmentation Crack Length Change** | Final crack length minus initial crack length, using the configured fraction of frames evaluated |
| **Segmentation Crack Length Maximum** | Maximum crack length across analyzed frames |
| **Segmentation Crack Width Maximum** | Maximum crack width across analyzed frames |

### Intensity Distribution Metrics
The Intensity Distribution module computes a intensity distribution histogram for all selected frames by the user. The histogram counts are normalized to provide a probability distribution of pixel intensity values, with intensity values with a probability below a user-defined noise threshold (defined [above](#intensity-distribution-settings)) being set to zero. From this intensity probability distribution, the kurtosis, median skewness (defined for a given histogram as $3 * \frac{\text{mean} - \text{median}}{\text{standard deviation}}$), and mode skewness (defined for a given histogram as $3 * \frac{\text{mean} - \text{mode}}{\text{standard deviation}}$). The following metrics are computed with respect to these definitions.
| Metric | Description |
| - | - |
| **Maximum Kurtosis** | The maximum kurtosis in the selected frames -- calculated by taking the top 10% of kurtosis values for the selected frames and averaging over those |
| **Maximum Median Skewness** | The maximum median skewness in the selected frames -- calculated in a similar manner to ***Maximum Kurtosis***. |
| **Maximum Mode Skewness** | The maximum mode skewness in the selected frames of the video -- calculated in a similar manner to ***Maximum Kurtosis***. |
| **Kurtosis Change** | The change in kurtosis between the beginning and the end of the video, calculated in a similar manner to the ***Maximum Island Area Change*** |
| **Median Skewness Change** | The change in the median skewness between the beginning and end of the video, calculated in a similar manner to the ***Kurtosis Change*** |
| **Mode Skewness Difference** | The change in the mode skewness between the beginning and end of the video, calculated in a similar manner to the ***Kurtosis Change*** |

## Output Files
The BARCODE program saves multiple outputs during the course of the analysis.
**Summary:** At the base level, the BARCODE program outputs a CSV file containing one row for each video channel analyzed in a given dataset. Each row lists the video file, analyzed channel, flags, and the summary metrics from the selected analysis branches. Any branch that is not used will have NaN (Not a Number) values populating the corresponding metrics for that branch.

  - **Flags:** The Flag column indicates potential issues with the reliability of some outputs. These flags should be checked when interpreting the barcode and summary CSV output.  
    - **Flag = 0:** No warning was detected.
    - **Flag = 1:** Dim file or channel - defined as videos where the first frame has a minimum pixel intensity greater than or equal to 2/e times the mean pixel intensity of the corresponding frame. This can affect analysis accuracy, and is therefore listed to give users insight as to the reliability of BARCODE's output for that file.
    - **Flag = 2:** Saturated file or channel - if the mode of the pixel intensity distribution coincides with the maximum intensity bin of the histogram in every analyzed frame. If even one analyzed frame does not satisfy this condition, no saturation warning is reported.  This can affect the accuracy of the Intensity Distribution branch, and is therefore only shown if the Intensity Distribution branch is used.
    - **Flag = 3:** Structural image autocorrelation (SIA) correlation length exceeds the field of view in at least one analyzed frame.
    - **Flag = 4:** Velocity correlation length exceeds the field of view in at least one analyzed frame.

- **Summary Barcode:** The BARCODE program can also output a visual representation of the data metrics described in the Summary file above. For each metric, this is done by normalizing the metric values using a combination of predetermined limits and the extrema values for a given metric to a 0-1 scale. These normalized values are then plotted using the Matplotlib color map "Plasma". These visualizations are separated by channel for ease of visualization.

- **Visualizations:** The program can also output graphs for visualization of the analysis performed by the modules. Binarization outputs before/after frame comparisons and largest island/void trends. Optical flow outputs representative flow fields and velocity-correlation plots. Intensity distribution outputs histograms of pixel intensities. Segmentation / Morphology can output mask overlays, labeled component images or video, edge density trends, centroid trajectories, masked-flow comparisons, and optional mechanics/crack evidence visualizations. Summary-level plots are saved in "Summary Graphs.png"; additional branch-specific diagnostics are saved inside the channel output folder.

- **Reduced Data Structures:** The program can also output reduced data structures used to perform the analysis, including binarized frames, structural autocorrelation, flow fields, intensity distributions, segmentation masks/components, masked-flow metrics, and mechanics/crack time-series outputs. These are saved as CSV files inside each channel output folder.

The visualizations and reduced data structures are saved in a folder titled ```{name of file} BARCODE Output```, saved in the same folder as the file, within a subfolder for each channel evaluated. The summary and barcode are saved in the root folder where the program is running.
