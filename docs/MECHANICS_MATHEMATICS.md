# Mathematics of the BARCODE Mechanics Extension

This document specifies the computations implemented in
`analysis/mechanics.py` and `analysis/segmentation.py`. The goal is to make every
reported quantity reproducible from the exported evidence arrays.

## 1. Notation and calibration

Let a single-channel image at frame (t) be

\[
I_t:\Omega\subset\mathbb{Z}^2\rightarrow\mathbb{R},
\qquad \Omega=\{0,\ldots,W-1\}\times\{0,\ldots,H-1\}.
\]

The binary foreground or crack mask is (M_t(x,y)\in\{0,1\}). The configured
physical pixel size is denoted by (s), with units such as micrometres per
pixel. Therefore:

- a pixel length becomes (s) physical length units;
- a pixel area becomes (s^2) physical area units;
- spatial derivatives use coordinate spacing (s).

If (s=1), all lengths and areas remain in pixels and pixels squared.

## 2. Robust 8-bit normalization

OpenCV segmentation and flow operate on an 8-bit image. A non-8-bit frame is
mapped using its 0.5th and 99.5th percentiles:

\[
I_8(x,y)=\operatorname{clip}\left(
255\frac{I(x,y)-P_{0.5}(I)}{P_{99.5}(I)-P_{0.5}(I)},0,255
\right).
\]

NaN and infinite values are replaced by zero first. If the two percentiles are
equal, the normalized frame is identically zero. An existing `uint8` frame is
copied without rescaling.

## 3. Segmentation

### 3.1 Adaptive-threshold MVP

The normalized frame is Gaussian-smoothed:

\[
I_b=G_{\sigma}*I_8.
\]

For an odd adaptive block size (B), let
(\mu_B(x,y)) be the Gaussian-weighted local mean. For bright foreground, the
local decision is

\[
M_{\mathrm{local}}(x,y)
=\mathbf{1}\!\left[I_b(x,y)>\mu_B(x,y)-C\right].
\]

For dark foreground (`invert_mask=True`), the inequality is reversed. The local
decision is gated by an Otsu global decision of the same polarity:

\[
M_0=M_{\mathrm{local}}\land M_{\mathrm{Otsu}}.
\]

This gate prevents both a uniform background and a uniform object interior from
being selected merely because of the adaptive offset (C).

Morphological opening removes isolated foreground noise:

\[
M\circ K=(M\ominus K)\oplus K,
\]

and morphological closing bridges small gaps:

\[
M\bullet K=(M\oplus K)\ominus K.
\]

Opening and closing are repeated according to their configured iteration
counts. Finally, a connected component (R_i) is retained only when

\[
|R_i|\ge A_{\min}.
\]

### 3.2 Optional model plug-in

A model plug-in is loaded from `module:function` and must compute either

\[
f(I_t)=M_t
\]

or

\[
f(I_t)=(M_t,C_t),
\]

where (C_t(x,y)\in[0,1]) is a confidence map. The mask and confidence map must
have the same height and width as the input frame. A missing confidence map is
replaced by ones. Small-component filtering is applied to the returned mask.

### 3.3 Legacy Canny segmentation

The backward-compatible mode computes Canny edges after Gaussian smoothing,
closes gaps morphologically, extracts exterior contours, rejects contours below
(A_{\min}), and fills the retained contours. Its confidence evidence is the
binary Canny edge map.

## 4. Segmentation confidence and QC

For adaptive segmentation, define the local variance and standard deviation:

\[
\sigma_B^2(x,y)=\mathbb{E}_B[I_b^2]-\mu_B^2,
\qquad
\sigma_B=\sqrt{\max(\sigma_B^2,0)}.
\]

The signed decision margin is

\[
m(x,y)=I_b(x,y)-\left(\mu_B(x,y)-C\right),
\]

with its sign reversed for dark foreground. Pixel confidence is

\[
C_t(x,y)=\operatorname{clip}\left(
\frac{|m(x,y)|}{\sigma_B(x,y)+|C|+10^{-6}},0,1
\right).
\]

Reported segmentation confidence is the mean over foreground pixels:

\[
\bar C_t=\frac{1}{|M_t|}\sum_{(x,y):M_t(x,y)=1}C_t(x,y).
\]

It is zero for an empty mask. The interpretable QC score starts at
(Q_t=\bar C_t), then applies multiplicative penalties:

\[
Q_t\leftarrow
Q_t\times
\begin{cases}
0.25,& |M_t|/(HW)<0.001\text{ or }>0.999,\\
1,&\text{otherwise},
\end{cases}
\]

\[
Q_t\leftarrow 0.8Q_t
\quad\text{if the mask touches the image border},
\]

\[
Q_t\leftarrow 0.5Q_t
\quad\text{if Canny fallback was required}.
\]

The final score is clipped to ([0,1]). The evidence also records foreground
fraction, connected-component count, border contact, and fallback status.

## 5. Edge detection and contours

Exterior contours are extracted directly from the binary mask. A one-pixel
boundary image (E_t) is drawn from those contours. Canny is used only when the
mask contains no exterior contour:

\[
E_t=
\begin{cases}
\partial M_t,&\partial M_t\ne\varnothing,\\
\operatorname{Canny}(I_t;T_{\mathrm{low}},T_{\mathrm{high}}),&\text{otherwise}.
\end{cases}
\]

Edge density is the boundary fraction of the field of view:

\[
\rho_{E,t}=\frac{\sum_{x,y}\mathbf{1}[E_t(x,y)>0]}{HW}.
\]

For contour (j), let its ordered points be
(p_0,\ldots,p_{n_j-1}), with (p_{n_j}=p_0). Its closed arc length is

\[
P_{j,t}=s\sum_{i=0}^{n_j-1}\|p_{i+1}-p_i\|_2.
\]

The implementation exports:

\[
P_{\mathrm{total},t}=\sum_jP_{j,t},\qquad
P_{\mathrm{mean},t}=\frac{1}{N_c}\sum_jP_{j,t},\qquad
P_{\max,t}=\max_jP_{j,t},
\]

along with contour count (N_c). All exterior contours contribute to these
statistics. In an NPZ file, contour (j) is reconstructed as

```python
points[offsets[j]:offsets[j + 1]]
```

where `points = contour_points` and `offsets = contour_offsets`.

## 6. Shape descriptors

Shape descriptors use the largest exterior contour. Let its area be (A_t) and
perimeter be (P_t):

\[
A_t=s^2\operatorname{ContourArea}(M_t),
\qquad
P_t=s\operatorname{ArcLength}(M_t).
\]

Circularity is

\[
\mathcal{C}_t=\frac{4\pi A_t}{P_t^2}.
\]

For fitted major and minor ellipse axes (a_t\ge b_t), elongation is

\[
\mathcal{E}_t=\frac{a_t}{b_t}.
\]

The orientation angle (\alpha_t) is the OpenCV fitted-ellipse angle. If fewer
than five contour points exist, a minimum-area rectangle supplies the axes and
angle.

For (q\in\{A,P,\mathcal C,\mathcal E\}), normalized change relative to the
first valid frame is

\[
\Delta q_t=\frac{q_t-q_0}{|q_0|}.
\]

If (q_0=0), the unscaled difference (q_t-q_0) is used. Orientation change is
wrapped to ([-90^\circ,90^\circ)) and normalized by (180^\circ):

\[
\Delta\alpha_t=
\frac{((\alpha_t-\alpha_0+90^\circ)\bmod180^\circ)-90^\circ}{180^\circ}.
\]

## 7. Signed contour curvature

Curvature uses the ordered points of the largest contour after multiplication
by pixel size (s). For periodic central differences,

\[
r'_i\approx\frac{r_{i+1}-r_{i-1}}{2},
\qquad
r''_i\approx r_{i+1}-2r_i+r_{i-1}.
\]

For (r=(x,y)), signed curvature is

\[
\kappa_i=
\frac{x'_iy''_i-y'_ix''_i}
{\left((x'_i)^2+(y'_i)^2\right)^{3/2}}.
\]

The denominator is protected at (10^{-12}). The exported signed map stores
(\kappa_i) at contour pixels. Summaries are mean signed curvature, mean
absolute curvature, and maximum absolute curvature. Curvature units are inverse
length.

## 8. Deformation and displacement

### 8.1 Dense optical flow

Farneback optical flow estimates a dense displacement between two analyzed
frames:

\[
u_t(x,y)=\left(u_x(x,y),u_y(x,y)\right)
=\operatorname{Farneback}(I_t,I_{t+\Delta t}).
\]

The vector field is multiplied by (s) to obtain physical displacement. This is
**displacement per analyzed frame pair**, not velocity; it is not divided by
frame time or frame step in the mechanics extension.

Displacement magnitude is

\[
d_t(x,y)=\sqrt{u_x^2+u_y^2}.
\]

The main results report the mean of frame-pair mean magnitudes and the maximum
observed magnitude.

### 8.2 Rigid phase-correlation registration

The alternative registration mode computes the normalized cross-power spectrum
of the two frames. The peak of its inverse Fourier transform gives the global
translation ((\delta_x,\delta_y)). A constant dense field is then exported:

\[
u_t(x,y)=s(\delta_x,\delta_y)\quad\forall(x,y)\in\Omega.
\]

## 9. Signed curl

For displacement (u=(u_x,u_y)), the out-of-plane signed curl is

\[
\omega=\frac{\partial u_y}{\partial x}
-\frac{\partial u_x}{\partial y}.
\]

Derivatives use NumPy finite differences with spatial spacing (s). The signed
map and its spatial mean are exported.

## 10. Strain tensors

Define the displacement gradient

\[
\nabla u=
\begin{bmatrix}
u_{x,x}&u_{x,y}\\
u_{y,x}&u_{y,y}
\end{bmatrix}.
\]

### 10.1 Small-strain tensor

The infinitesimal strain tensor is

\[
\varepsilon=\frac{1}{2}\left(\nabla u+(\nabla u)^T\right),
\]

so

\[
\varepsilon_{xx}=u_{x,x},\qquad
\varepsilon_{yy}=u_{y,y},\qquad
\varepsilon_{xy}=\frac{1}{2}(u_{x,y}+u_{y,x}).
\]

### 10.2 Finite-strain tensor

The finite option uses the Green-Lagrange tensor

\[
E=\frac{1}{2}\left(F^TF-I\right),
\qquad F=I+\nabla u.
\]

Its implemented components are

\[
E_{xx}=u_{x,x}+\frac{1}{2}\left(u_{x,x}^2+u_{y,x}^2\right),
\]

\[
E_{yy}=u_{y,y}+\frac{1}{2}\left(u_{x,y}^2+u_{y,y}^2\right),
\]

\[
E_{xy}=\frac{1}{2}\left(
u_{x,y}+u_{y,x}+u_{x,x}u_{x,y}+u_{y,x}u_{y,y}
\right).
\]

The three signed component maps and their spatial means are exported.

## 11. Crack skeleton and graph length

The selected crack mask is skeletonized to a one-pixel-wide set (S_t). The
user may skeletonize the foreground mask or its inverse. Each skeleton pixel is
a graph vertex. Vertices are connected to their 8-neighbors with weight

\[
w(p,q)=
\begin{cases}
s,&\text{horizontal or vertical neighbors},\\
s\sqrt{2},&\text{diagonal neighbors}.
\end{cases}
\]

Because each undirected edge is encountered from both endpoints, total crack
length is

\[
L_t=\frac{1}{2}\sum_{p\in S_t}\sum_{q\in N(p)}w(p,q).
\]

A tip has graph degree one. A branch pixel has degree greater than two. Adjacent
branch pixels are grouped into a single branch region by 8-connected-component
labeling, preventing a multi-pixel junction from being counted repeatedly.

The longest path is the maximum weighted Dijkstra distance obtained from all
tips. For a closed skeleton with no tips, the implementation reports the
farthest distance from one arbitrary skeleton vertex; this is not the full loop
circumference.

## 12. Crack width

Let (D_t(x,y)) be the Euclidean distance transform inside the binary crack
mask: the distance from an interior pixel to the nearest background pixel.
Local crack width is sampled only at skeleton pixels:

\[
W_t(p)=2sD_t(p),\qquad p\in S_t.
\]

This is the diameter of the maximal disk centered at the skeleton pixel and
contained in the segmented crack. The exported width map is zero away from the
skeleton. Frame summaries are

\[
\bar W_t=\frac{1}{|S_t|}\sum_{p\in S_t}W_t(p),
\qquad
W_{\max,t}=\max_{p\in S_t}W_t(p).
\]

Width accuracy therefore depends directly on mask accuracy and on the skeleton
being near the medial axis. Junctions, merged cracks, and pixel discretization
can inflate local width.

## 13. Crack-tip tracking

Tips are matched between consecutive analyzed frames by nearest neighbor. For a
tip (p_i^t), its displacement is

\[
d_i^{\mathrm{tip}}=s\min_j\|p_i^t-p_j^{t+\Delta t}\|_2.
\]

The pair summary contains mean and maximum nearest-tip displacement. This is a
local nearest-neighbor association, not a globally constrained multi-object
tracker; tips can switch identity when cracks cross or branch.

## 14. Temporal and dataset summaries

For analyzed frame indices (t_1,\ldots,t_N), the main BARCODE row contains:

- mean and maximum edge density;
- edge-density ratio
  (\overline{\rho}_{E,\mathrm{final}}/
  \overline{\rho}_{E,\mathrm{initial}}), where the configured evaluation
  fraction selects frames at each end;
- mean segmented-area fraction and mean component count;
- mean confidence and mean QC score;
- mean frame-level mean absolute curvature and maximum observed absolute
  curvature;
- the final normalized delta for each shape descriptor;
- mean displacement and maximum displacement;
- mean signed curl and mean signed strain components;
- maximum crack length and normalized crack-length change

  \[
  \Delta L=(L_N-L_1)/|L_1|;
  \]
- maximum branch count and mean tip displacement;
- mean frame-level contour length and maximum frame-level total contour length;
- mean frame-level crack width and maximum observed crack width.

Undefined divisions, empty masks, and unavailable frame-pair measurements are
reported as NaN unless the computation has a natural zero, such as zero crack
length for an empty skeleton.

## 15. Exported evidence

With **Save Reduced Data Structures** enabled, each analyzed frame archive
contains numeric arrays:

- `mask`
- `confidence`
- `boundary`
- `contour_points` and `contour_offsets`
- `curvature`
- `crack_skeleton`
- `crack_width`

Each analyzed frame-pair archive contains:

- `displacement`
- `curl`
- `strain_xx`
- `strain_yy`
- `strain_xy`

`MechanicsTimeSeries.csv` contains the scalar frame measurements. The NPZ files
contain only numeric arrays and can be opened without pickle:

```python
import numpy as np

with np.load("Mechanics Evidence/frame_00000.npz", allow_pickle=False) as data:
    width = data["crack_width"]
    points = data["contour_points"]
```

## 16. Interpretation limits

1. The segmentation mask is the measurement foundation. Edge, contour, crack,
   width, and shape errors inherit segmentation errors.
2. Adaptive threshold parameters should be calibrated using the live preview on
   representative frames.
3. Contour curvature is sensitive to pixel stair-stepping because the MVP uses
   finite differences without spline smoothing.
4. Shape descriptors refer to the largest exterior contour; contour-length
   statistics include every exterior contour.
5. Farneback results are pairwise displacements, not material velocities.
6. Strain computed from optical flow is meaningful only when the displacement
   field is spatially resolved and the calibration is correct.
7. Distance-transform width is appropriate for resolved, segmented cracks; it
   is not a substitute for a sub-pixel optical-width model.
8. Nearest-neighbor tip tracking is intentionally interpretable but may be
   ambiguous at branching or merging events.

## 17. Test correspondence

The deterministic tests in `tests/test_mechanics.py` cover:

- adaptive and plug-in segmentation contracts;
- contour-derived boundaries and Canny fallback;
- contour extraction and arc length;
- signed curvature and shape descriptors;
- normalized shape change;
- known rigid translation;
- analytic signed curl;
- analytic small- and finite-strain fields;
- skeleton length, branches, tips, width, and tip tracking;
- QC penalties.

`tests/test_segmentation.py` verifies end-to-end video analysis, NPZ evidence,
CSV round trips, and YAML configuration round trips.
