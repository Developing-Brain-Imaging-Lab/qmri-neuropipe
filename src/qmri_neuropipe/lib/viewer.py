import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional, List
import warnings

# Attempt to import optional viewer dependencies
try:
    import napari
    from magicgui import magicgui
    from qtpy.QtWidgets import (
        QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, 
        QScrollArea, QSplitter
    )
    from qtpy.QtCore import Qt
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False

from .synthetic import (
    synthetic_se, synthetic_spgr, synthetic_ir, 
    synthetic_ssfp, synthetic_mprage, synthetic_flair, synthetic_dwi
)

def _load_nifti(path: Path):
    if not path or not path.exists():
        return None
    try:
        img = nib.load(path)
        # Reorient to standard RAS+ to behave like mrview
        img_canonical = nib.as_closest_canonical(img)
        data = np.asanyarray(img_canonical.dataobj).squeeze()
        
        # Extract basic header info for the UI
        hdr = img_canonical.header
        meta = {
            "Filename": path.name,
            "Dimensions": str(img_canonical.shape),
            "Voxel Size (mm)": str(hdr.get_zooms()),
            "Datatype": str(hdr.get_data_dtype()),
            "qform_code": str(hdr.get_qform(coded=True)[1]),
            "sform_code": str(hdr.get_sform(coded=True)[1]),
        }
        # Dump entire affine and select header fields as raw text representation
        raw_header_text = str(hdr)
        meta["raw_header"] = raw_header_text
        
        return {"data": data, "meta": meta, "name": path.name}
    except Exception as e:
        warnings.warn(f"Failed to load {path}: {e}")
        return None

def launch_viewer(
    images: List[Path] = None,
    t1_path: Optional[Path] = None,
    t2_path: Optional[Path] = None,
    m0_path: Optional[Path] = None,
    adc_path: Optional[Path] = None
):
    """
    Launch the Napari-based NIfTI Viewer.
    """
    if not NAPARI_AVAILABLE:
        raise ImportError(
            "Napari and magicgui are required for the viewer. "
            "Install with: pip install -e .[viewer]"
        )

    viewer = napari.Viewer(title="qMRI Neuropipe Viewer")

    # Load generic images
    if images:
        for img_p in images:
            img_dict = _load_nifti(img_p)
            if img_dict is not None:
                viewer.add_image(
                    img_dict["data"], 
                    name=img_dict["name"], 
                    colormap="gray", 
                    blending="additive",
                    metadata={"nifti_header": img_dict["meta"]}
                )

    # Load quantitative maps for synthetic generation
    t1_dict = _load_nifti(t1_path)
    t2_dict = _load_nifti(t2_path)
    m0_dict = _load_nifti(m0_path)
    adc_dict = _load_nifti(adc_path)

    t1_map = t1_dict["data"] if t1_dict else None
    t2_map = t2_dict["data"] if t2_dict else None
    m0_map = m0_dict["data"] if m0_dict else None
    adc_map = adc_dict["data"] if adc_dict else None

    # Automatically generate an M0 placeholder if T1/T2 are present but M0 is not
    if m0_map is None and (t1_map is not None or t2_map is not None):
        ref_map = t1_map if t1_map is not None else t2_map
        m0_map = np.ones_like(ref_map) * 1000.0  # arbitrary proton density
        m0_dict = {"meta": {"Warning": "Auto-filled dummy M0 map (Proton Density set to 1000)"}}

    if t1_map is not None: viewer.add_image(t1_map, name="T1 Map", colormap="magma", visible=False, metadata={"nifti_header": t1_dict["meta"]})
    if t2_map is not None: viewer.add_image(t2_map, name="T2 Map", colormap="viridis", visible=False, metadata={"nifti_header": t2_dict["meta"]})
    if m0_map is not None: viewer.add_image(m0_map, name="M0 Map", colormap="gray", visible=False, metadata={"nifti_header": m0_dict["meta"]})
    if adc_map is not None: viewer.add_image(adc_map, name="ADC/MD Map", colormap="turbo", visible=False, metadata={"nifti_header": adc_dict["meta"]})

    has_relaxometry = t1_map is not None and m0_map is not None
    seq_choices = ["SE", "SPGR (FLASH)", "IR", "FLAIR", "bSSFP", "MPRAGE"] if has_relaxometry else ["Generic View Only"]
    if adc_map is not None:
        seq_choices.append("DWI")

    # Create the Interactive Synthetic MRI Widget
    @magicgui(
        auto_call=True,
        sequence={"choices": seq_choices, "label": "Sequence Type"},
        tr={"widget_type": "Slider", "max": 10000, "min": 1, "step": 10, "label": "TR (ms)"},
        te={"widget_type": "Slider", "max": 500, "min": 1, "step": 1, "label": "TE (ms)"},
        ti={"widget_type": "Slider", "max": 5000, "min": 1, "step": 10, "label": "TI (ms)"},
        fa={"widget_type": "Slider", "max": 180, "min": 1, "step": 1, "label": "Flip Angle (deg)"},
        bval={"widget_type": "Slider", "max": 5000, "min": 0, "step": 50, "label": "b-value (s/mm²)"}
    )
    def synthetic_control(
        sequence: str = seq_choices[0],
        tr: int = 1000,
        te: int = 10,
        ti: int = 400,
        fa: int = 90,
        bval: int = 1000
    ):
        if not has_relaxometry and sequence != "DWI":
            return
            
        # Default placeholder T1/T2 maps if one is missing but requested
        t1 = t1_map if t1_map is not None else np.zeros_like(m0_map)
        t2 = t2_map if t2_map is not None else np.zeros_like(m0_map)
        
        # Calculate signal
        if sequence == "SE":
            signal = synthetic_se(t1, t2, m0_map, tr, te)
        elif sequence == "SPGR (FLASH)":
            signal = synthetic_spgr(t1, m0_map, tr, fa)
        elif sequence == "IR":
            signal = synthetic_ir(t1, t2, m0_map, tr, te, ti)
        elif sequence == "FLAIR":
            signal = synthetic_flair(t1, t2, m0_map, tr, te, ti)
        elif sequence == "bSSFP":
            signal = synthetic_ssfp(t1, t2, m0_map, tr, fa)
        elif sequence == "MPRAGE":
            signal = synthetic_mprage(t1, m0_map, tr, ti, fa)
        elif sequence == "DWI":
            if adc_map is not None:
                s0 = m0_map if m0_map is not None else np.ones_like(adc_map) * 1000.0
                signal = synthetic_dwi(adc_map, s0, bval)
            else:
                signal = np.zeros_like(m0_map) if m0_map is not None else np.zeros((10,10,10))
        else:
            return

        # Update or add layer
        layer_name = f"Synthetic {sequence}"
        if layer_name in viewer.layers:
            viewer.layers[layer_name].data = signal
        else:
            viewer.add_image(signal, name=layer_name, colormap="gray")
            
        # Optional: Auto-adjust contrast limits on sequence switch (basic heuristic)
        if layer_name in viewer.layers:
            p2, p98 = np.percentile(signal[signal > 0], (2, 98)) if np.any(signal > 0) else (0, 1)
            viewer.layers[layer_name].contrast_limits = (0, p98 * 1.2 if p98 > 0 else 1)

    viewer.window.add_dock_widget(synthetic_control, name="Synthetic Parameters", area="right")

    # Custom Layout Tools Dock Widget
    class LayoutToolsWidget(QWidget):
        def __init__(self, viewer: napari.Viewer):
            super().__init__()
            self.viewer = viewer
            layout = QVBoxLayout()
            
            lbl = QLabel("Neuroimaging Layout Options")
            lbl.setStyleSheet("font-weight: bold;")
            layout.addWidget(lbl)
            
            btn_grid = QPushButton("Toggle Lightbox (Grid) Mode")
            btn_grid.clicked.connect(self.toggle_grid)
            layout.addWidget(btn_grid)
            
            btn_3d = QPushButton("Toggle 3D Volume View")
            btn_3d.clicked.connect(self.toggle_3d)
            layout.addWidget(btn_3d)
            
            # 3-Plane Orthogonal Button
            btn_ortho = QPushButton("Launch 3-Plane Orthogonal View")
            btn_ortho.clicked.connect(self.launch_ortho)
            layout.addWidget(btn_ortho)
            
            # Custom robust NIfTI loader to bypass buggy napari plugins
            btn_load = QPushButton("Load NIfTI Image...")
            btn_load.clicked.connect(self.load_image)
            layout.addWidget(btn_load)
            
            # Header Viewer Button
            btn_header = QPushButton("Inspect Image Header")
            btn_header.clicked.connect(self.show_header)
            layout.addWidget(btn_header)

            self.setLayout(layout)
            
        def toggle_grid(self):
            # Toggle grid layout
            curr = self.viewer.grid.enabled
            self.viewer.grid.enabled = not curr
            if self.viewer.grid.enabled:
                self.viewer.grid.shape = (-1, 3) # Arbitrary neat grid
                
        def toggle_3d(self):
            # Toggle 2D/3D
            curr = self.viewer.dims.ndisplay
            self.viewer.dims.ndisplay = 3 if curr == 2 else 2
            
        def launch_ortho(self):
            # Embed the 3 planes directly into the main window's central widget area.
            embed_orthogonal_view(self.viewer)

        def load_image(self):
            # Show file dialog
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Select NIfTI Image", "", "NIfTI Files (*.nii *.nii.gz);;All Files (*)"
            )
            if filepath:
                img_dict = _load_nifti(Path(filepath))
                if img_dict is not None:
                    self.viewer.add_image(
                        img_dict["data"], 
                        name=img_dict["name"], 
                        colormap="gray", 
                        blending="additive",
                        metadata={"nifti_header": img_dict["meta"]}
                    )
                    
        def show_header(self):
            active_layers = list(self.viewer.layers.selection)
            if not active_layers:
                QMessageBox.warning(self, "No Selection", "Please select a layer to view its header.")
                return
            
            # Only show first selected layer
            layer = active_layers[0]
            header_meta = layer.metadata.get("nifti_header")
            
            if not header_meta:
                QMessageBox.information(self, "No Header", f"No spatial NIfTI header found for layer '{layer.name}'. "
                                        "(May be a dynamically generated array).")
                return
                
            # Build string output
            lines = [f"<b>Layer:</b> {layer.name}<br>"]
            for k, v in header_meta.items():
                if k == "raw_header":
                    continue
                lines.append(f"<b>{k}:</b> {v}")
                
            summary = "<br>".join(lines)
            
            # Message box with scrollable detailed text
            msg = QMessageBox(self)
            msg.setWindowTitle("NIfTI Header Metadata")
            msg.setTextFormat(Qt.RichText)
            msg.setText(summary)
            
            if "raw_header" in header_meta:
                msg.setDetailedText(header_meta["raw_header"])
                
            msg.exec_()

    layout_tools = LayoutToolsWidget(viewer)
    viewer.window.add_dock_widget(layout_tools, name="Layout Tools", area="left")

    # Start the application loop
    napari.run()

def embed_orthogonal_view(main_viewer: "napari.Viewer"):
    """
    Replaces the main Napari canvas with a 3-plane synchronized orthogonal layout 
    (Sagittal, Coronal, Axial).
    """
    if not NAPARI_AVAILABLE: return
    
    # Create 3 sub-viewers
    viewer_ax = napari.Viewer(title="Axial", show=False)
    viewer_cor = napari.Viewer(title="Coronal", show=False)
    viewer_sag = napari.Viewer(title="Sagittal", show=False)
    
    viewers = [viewer_ax, viewer_cor, viewer_sag]
    
    # Hide axis and UI elements on sub-viewers to save space
    for v in viewers:
        v.window.qt_viewer.dockLayerControls.hide()
        v.window.qt_viewer.dockLayerList.hide()
        v.scale_bar.visible = True
        v.axes.visible = True
    
    # Set display axes (Assuming RAS+ standard: 0=Sagittal, 1=Coronal, 2=Axial)
    viewer_ax.dims.order = (0, 1, 2)
    viewer_cor.dims.order = (0, 2, 1)
    viewer_sag.dims.order = (1, 2, 0)
    
    # Replicate active layers from the main viewer
    for layer in main_viewer.layers:
        if isinstance(layer, napari.layers.Image):
            for v in viewers:
                v.add_image(
                    layer.data, name=layer.name, colormap=layer.colormap, 
                    blending=layer.blending, contrast_limits=layer.contrast_limits,
                    visible=layer.visible
                )
                
    # Keep the main viewer's layers synchronized with the sub-viewers
    def mirror_layer_changes(event):
        """Update sub-viewers when layers are added/removed from main viewer."""
        # Simple implementation: re-sync all layers (could be optimized)
        for v in viewers:
            v.layers.clear()
        for layer in main_viewer.layers:
            if isinstance(layer, napari.layers.Image):
                for v in viewers:
                    v.add_image(
                        layer.data, name=layer.name, colormap=layer.colormap, 
                        blending=layer.blending, contrast_limits=layer.contrast_limits,
                        visible=layer.visible
                    )
    
    main_viewer.layers.events.inserted.connect(mirror_layer_changes)
    main_viewer.layers.events.removed.connect(mirror_layer_changes)
    
    # Link cursor positions (crosshairs)
    def link_cursors(source_viewer, target_viewers):
        def on_cursor_move(event):
            pos = source_viewer.cursor.position
            for t in target_viewers:
                t.cursor.position = pos
        source_viewer.cursor.events.position.connect(on_cursor_move)
        
    link_cursors(viewer_ax, [viewer_cor, viewer_sag])
    link_cursors(viewer_cor, [viewer_ax, viewer_sag])
    link_cursors(viewer_sag, [viewer_ax, viewer_cor])
    link_cursors(main_viewer, viewers)
    for v in viewers:
        link_cursors(v, [main_viewer])
    
    # Construct the UI Splitter
    from qtpy.QtWidgets import QSplitter
    from qtpy.QtCore import Qt
    
    splitter = QSplitter(Qt.Horizontal)
    
    # Add the main viewer (for 3D volume, or keeping original view) or just the 3 planes
    # User requested replacing the main display. We will hide the main viewer canvas
    # and put our splitter in its place.
    main_qt_viewer = main_viewer.window.qt_viewer
    
    # The layout structure:
    # main_qt_viewer is a QWidget containing docks and the central canvas.
    # The central canvas is a custom widget (QSplitter -> viewerWidget)
    # We will grab the main window's central widget layout and inject our splitter.
    
    splitter.addWidget(viewer_sag.window.qt_viewer)
    splitter.addWidget(viewer_cor.window.qt_viewer)
    splitter.addWidget(viewer_ax.window.qt_viewer)
    
    # Ensure they resize evenly
    splitter.setSizes([400, 400, 400])
    
    # Find the main QMainWindow of napari
    main_window = main_qt_viewer.window()
    
    # Replace the central widget of the main window with our new 3-pane splitter
    # This completely overrides the default single-canvas `qt_viewer` display
    # but keeps the docked widgets (Layer list, synthetic controls, etc.) intact.
    
    # To keep the original main_viewer functionality accessible via scripts/docks,
    # we don't destroy it, just swap the UI element showing the image canvas out.
    # The original widget might be inside a QSplitter or layout in the central widget.
    
    central = main_window.centralWidget()
    if central:
        old_layout = central.layout()
        if old_layout:
            # Hide the old viewer canvas. (usually child at index 0)
            for i in reversed(range(old_layout.count())): 
                widget = old_layout.itemAt(i).widget()
                if widget:
                    widget.hide()
            
            # Add the new 3-plane splitter to the layout
            old_layout.addWidget(splitter)
            
    # Keep reference to avoid garbage collection
    main_viewer._ortho_splitters = viewers
