"""Interactive settings and live preview for edge segmentation."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from analysis.segmentation import segment_frame
from gui.config import BarcodeConfigGUI, InputConfigGUI, PreviewConfigGUI
from utils.gui import create_popup


def create_segmentation_frame(
    parent,
    config: BarcodeConfigGUI,
    preview_config: PreviewConfigGUI,
    input_config: InputConfigGUI,
):
    """Create segmentation controls with original/edge/mask live previews."""
    frame = ttk.Frame(parent)
    settings = config.segmentation_parameters
    preview = preview_config
    inputs = input_config
    row = 0

    def add_scale(label_text, variable, minimum, maximum, help_text):
        nonlocal row
        label = tk.Label(frame, text=f"{label_text}:")
        label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        create_popup(frame, help_text, row, label)
        tk.Scale(
            frame,
            from_=minimum,
            to=maximum,
            resolution=1,
            orient="horizontal",
            variable=variable,
            length=360,
        ).grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=2)
        row += 1

    add_scale(
        "Canny lower threshold",
        settings.canny_low,
        0,
        254,
        "Lower hysteresis threshold. Reduce it to retain weaker boundaries.",
    )
    add_scale(
        "Canny upper threshold",
        settings.canny_high,
        1,
        255,
        "Upper hysteresis threshold. Strong boundaries above it are retained.",
    )

    controls = [
        ("Gaussian blur kernel", settings.blur_kernel, tuple(range(1, 32, 2)),
         "Positive odd denoising kernel used before edge detection."),
        ("Closing kernel", settings.close_kernel, tuple(range(1, 32)),
         "Morphological kernel used to bridge nearby edge gaps."),
        ("Closing iterations", settings.close_iterations, tuple(range(0, 11)),
         "Number of edge-closing passes. Higher values join larger gaps."),
    ]
    for text, variable, values, help_text in controls:
        label = tk.Label(frame, text=f"{text}:")
        label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        create_popup(frame, help_text, row, label)
        ttk.Combobox(
            frame, textvariable=variable, values=values, state="readonly", width=8
        ).grid(row=row, column=1, sticky="w", padx=5, pady=5)
        row += 1

    minimum_label = tk.Label(frame, text="Minimum segment area (px):")
    minimum_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
    create_popup(
        frame,
        "Closed contours smaller than this pixel area are removed as noise.",
        row,
        minimum_label,
    )
    ttk.Spinbox(
        frame,
        from_=0,
        to=1_000_000,
        increment=10,
        textvariable=settings.minimum_segment_area,
        width=10,
    ).grid(row=row, column=1, sticky="w", padx=5, pady=5)
    row += 1

    frame_label = tk.Label(frame, text="Preview frame:")
    frame_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
    frame_slider = tk.Scale(
        frame,
        from_=0,
        to=0,
        resolution=1,
        orient="horizontal",
        variable=preview.preview_frame_number,
        length=360,
    )
    frame_slider.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=2)
    row += 1

    tk.Label(frame, text="Preview file:").grid(
        row=row, column=0, sticky="w", padx=5, pady=5
    )
    sample_files = ttk.Combobox(
        frame, textvariable=preview.sample_file, state="disabled", width=45
    )
    sample_files.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
    row += 1

    tk.Label(frame, text="Original").grid(row=row, column=0, pady=(10, 2))
    tk.Label(frame, text="Canny edges").grid(row=row, column=1, pady=(10, 2))
    tk.Label(frame, text="Filled segments").grid(row=row, column=2, pady=(10, 2))
    row += 1

    root = parent.winfo_toplevel()
    red, green, blue = root.winfo_rgb(root.cget("bg"))
    background = (red / 65535, green / 65535, blue / 65535)
    axes = []
    canvases = []
    for column in range(3):
        figure = Figure(figsize=(3, 3), facecolor=background)
        axis = figure.add_subplot(111)
        axis.axis("off")
        canvas = FigureCanvasTkAgg(figure, master=frame)
        canvas.get_tk_widget().grid(row=row, column=column, padx=4, pady=4)
        axes.append(axis)
        canvases.append(canvas)
    row += 1

    status = tk.Label(
        frame,
        text="Choose a file in Execution Settings to start the live preview.",
        anchor="w",
    )
    status.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    row += 1

    # These affect batch processing but do not change a single-frame preview.
    batch_fields = [
        ("Frame step", settings.frame_step, 1, 100, 1),
        ("Fraction of frames evaluated", settings.percentage_frames_evaluated, 0.01, 0.5, 0.01),
    ]
    for text, variable, minimum, maximum, increment in batch_fields:
        tk.Label(frame, text=f"{text}:").grid(
            row=row, column=0, sticky="w", padx=5, pady=5
        )
        ttk.Spinbox(
            frame,
            from_=minimum,
            to=maximum,
            increment=increment,
            textvariable=variable,
            width=10,
        ).grid(row=row, column=1, sticky="w", padx=5, pady=5)
        row += 1

    preview_data = {"frame": None, "job": None}

    def clear_preview(message):
        for axis, canvas in zip(axes, canvases):
            axis.clear()
            axis.axis("off")
            canvas.draw_idle()
        status.config(text=message)

    def render_preview():
        preview_data["job"] = None
        image = preview_data["frame"]
        if image is None:
            clear_preview("Choose a file in Execution Settings to start the live preview.")
            return
        try:
            current_config = settings.config
            edges, mask, measurements = segment_frame(image, current_config)
        except (tk.TclError, ValueError) as error:
            clear_preview(f"Adjust the segmentation parameters: {error}")
            return

        display_scale = max(1, int(np.ceil(max(image.shape) / 500)))
        images = (image, edges, mask)
        for axis, canvas, data in zip(axes, canvases, images):
            axis.clear()
            axis.imshow(data[::display_scale, ::display_scale], cmap="gray", interpolation="nearest")
            axis.axis("off")
            canvas.figure.tight_layout(pad=0.1)
            canvas.draw_idle()
        status.config(
            text=(
                f"Edges: {measurements['edge_density']:.2%} of FOV    "
                f"Segmented: {measurements['segmented_area']:.2%} of FOV    "
                f"Segments: {measurements['segment_count']}"
            )
        )

    def schedule_preview(*_args):
        # Debouncing keeps slider movement responsive on large microscopy frames.
        if preview_data["job"] is not None:
            frame.after_cancel(preview_data["job"])
        preview_data["job"] = frame.after(75, render_preview)

    def load_preview_frame(*_args):
        path = preview.sample_file.get() if inputs.mode.get() == "dir" else inputs.file_path.get()
        if inputs.mode.get() != "dir" and path and preview.sample_file.get() != path:
            preview.sample_file.set(path)
            return
        if not path:
            preview_data["frame"] = None
            schedule_preview()
            return
        try:
            data = preview.sample_preview
            if data is None or len(data) == 0:
                raise ValueError("the selected file contains no preview frames")
            frame_slider.config(to=max(0, len(data) - 1))
            frame_index = min(max(0, preview.preview_frame_number.get()), len(data) - 1)
            channel = 0 if config.channels.parse_all_channels.get() else config.channels.selected_channel.get()
            channel %= data.shape[3]
            preview_data["frame"] = data[frame_index, :, :, channel]
            schedule_preview()
        except Exception as error:
            preview_data["frame"] = None
            clear_preview(f"Unable to load preview from {path}: {error}")

    def update_sample_file_options(*_args):
        directory = inputs.dir_path.get()
        if directory and os.path.isdir(directory):
            files = sorted(
                os.path.join(root_path, filename)
                for root_path, _, filenames in os.walk(directory)
                for filename in filenames
                if filename.lower().endswith((".tif", ".tiff", ".nd2", ".avi", ".mp4"))
            )
            sample_files["values"] = files
            sample_files.config(state="readonly" if files else "disabled")
            if files and preview.sample_file.get() not in files:
                preview.sample_file.set(files[0])
        else:
            sample_files.set("")
            sample_files["values"] = []
            sample_files.config(state="disabled")

    inputs.file_path.trace_add("write", load_preview_frame)
    inputs.dir_path.trace_add("write", update_sample_file_options)
    preview.sample_file.trace_add("write", load_preview_frame)
    preview.preview_frame_number.trace_add("write", load_preview_frame)
    config.channels.selected_channel.trace_add("write", load_preview_frame)
    config.channels.parse_all_channels.trace_add("write", load_preview_frame)
    for variable in (
        settings.canny_low,
        settings.canny_high,
        settings.blur_kernel,
        settings.close_kernel,
        settings.close_iterations,
        settings.minimum_segment_area,
    ):
        variable.trace_add("write", schedule_preview)

    update_sample_file_options()
    load_preview_frame()
    return frame
