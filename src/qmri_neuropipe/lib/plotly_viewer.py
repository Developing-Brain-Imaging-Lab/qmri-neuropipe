import os
import sys
import threading
import time
import socket
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional, List
import warnings

# Attempt to import dependencies
try:
    import dash
    from dash import dcc, html, Input, Output, State, ctx
    import plotly.graph_objects as go
    import webview
    import base64
    import tempfile
    DASH_AVAILABLE = True
    IMPORT_ERROR_MSG = ""
except ImportError as e:
    DASH_AVAILABLE = False
    IMPORT_ERROR_MSG = str(e)

from .synthetic import (
    synthetic_se, synthetic_spgr, synthetic_ir, 
    synthetic_ssfp, synthetic_mprage, synthetic_flair, synthetic_dwi
)

def _load_nifti(path: Path):
    if not path or not path.exists():
        return None
    try:
        img = nib.load(path)
        img_canonical = nib.as_closest_canonical(img)
        data = np.asanyarray(img_canonical.dataobj).squeeze()
        return data
    except Exception as e:
        warnings.warn(f"Failed to load {path}: {e}")
        return None

def launch_viewer(
    images: List[Path] = None,
    t1_path: Optional[Path] = None,
    t2_path: Optional[Path] = None,
    m0_path: Optional[Path] = None,
    adc_path: Optional[Path] = None,
    port: int = 8050
):
    if not DASH_AVAILABLE:
        raise ImportError(
            f"Dash, Plotly, and PyWebView are required for the viewer. "
            f"Install with: pip install dash plotly pywebview\nError details: {IMPORT_ERROR_MSG}"
        )

    # Load data
    t1_map = _load_nifti(t1_path)
    t2_map = _load_nifti(t2_path)
    m0_map = _load_nifti(m0_path)
    adc_map = _load_nifti(adc_path)
    
    # Generic images handling
    base_img = None
    if images and len(images) > 0:
        base_img = _load_nifti(images[0])
    
    # Store data globally for callbacks (simple approach for single-user desktop app)
    global_store = {
        't1': t1_map,
        't2': t2_map,
        'm0': m0_map,
        'adc': adc_map,
        'base': base_img,
        'ref': None,
        'shape': (100, 100, 100) # Default empty shape
    }
    
    def update_ref_shape():
        ref = global_store['t1'] if global_store['t1'] is not None else \
              (global_store['t2'] if global_store['t2'] is not None else \
              (global_store['adc'] if global_store['adc'] is not None else global_store['base']))
        
        global_store['ref'] = ref
        if ref is not None:
            global_store['shape'] = ref.shape
            
    update_ref_shape()
    
    if global_store['m0'] is None and (global_store['t1'] is not None or global_store['t2'] is not None):
        global_store['m0'] = np.ones_like(global_store['ref']) * 1000.0

    app = dash.Dash(__name__, title="qMRI Neuropipe Viewer", suppress_callback_exceptions=True)
    
    def get_seq_choices():
        c = []
        if global_store['t1'] is not None and global_store['m0'] is not None:
            c.extend(["Generic", "SE", "SPGR (FLASH)", "IR", "FLAIR", "bSSFP", "MPRAGE"])
        if global_store['adc'] is not None:
            c.append("DWI")
        if not c:
            c = ["Generic"]
        return c
    
    seq_choices = get_seq_choices()
    
    # UI Layout
    # Create reusable upload component
    def make_upload(id_name, label):
        return dcc.Upload(
            id=id_name,
            children=html.Div([
                f'Drag and Drop or ',
                html.A('Select '+label)
            ]),
            style={
                'width': '100%',
                'height': '40px',
                'lineHeight': '40px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '5px 0',
                'color': '#aaa',
                'cursor': 'pointer'
            },
            multiple=False
        )

    app.layout = html.Div([
        # Hidden div to trigger data refreshes
        html.Div(id='data-refresh-trigger', style={'display': 'none'}),
        
        html.Div([
            # Controls Sidebar
            html.Div([
                html.H4("Data Management", style={'color': 'white', 'marginTop': '0px'}),
                make_upload('upload-generic', 'Generic Image'),
                make_upload('upload-t1', 'T1 Map'),
                make_upload('upload-t2', 'T2 Map'),
                make_upload('upload-m0', 'M0 Map'),
                make_upload('upload-adc', 'ADC Map'),
                html.Hr(style={'borderColor': '#555'}),
                
                html.H4("Controls", style={'color': 'white'}),
                html.Label("Sequence", style={'color': '#aaa', 'fontSize': '14px'}),
                dcc.Dropdown(
                    id='sequence-dropdown',
                    options=[],
                    value="Generic",
                    style={'marginBottom': '15px', 'color': 'black'},
                    clearable=False
                ),
                
                html.Label("Sagittal Slice (X)", style={'color': '#aaa', 'fontSize': '14px'}),
                dcc.Slider(0, 100, 1, value=50, id='slice-x-slider', tooltip={"placement": "bottom", "always_visible": True}),
                html.Label("Coronal Slice (Y)", style={'color': '#aaa', 'fontSize': '14px'}),
                dcc.Slider(0, 100, 1, value=50, id='slice-y-slider', tooltip={"placement": "bottom", "always_visible": True}),
                html.Label("Axial Slice (Z)", style={'color': '#aaa', 'fontSize': '14px'}),
                dcc.Slider(0, 100, 1, value=50, id='slice-z-slider', tooltip={"placement": "bottom", "always_visible": True}),
                
                html.Hr(style={'borderColor': '#555'}),
                html.H4("Synthetic Parameters", style={'color': 'white'}),
                html.Label("TR (ms)", style={'color': '#aaa', 'fontSize': '14px'}),
                dcc.Slider(1, 10000, 10, value=1000, id='tr-slider', tooltip={"placement": "bottom", "always_visible": False}),
                html.Label("TE (ms)", style={'color': '#aaa', 'fontSize': '14px'}),
                dcc.Slider(1, 500, 1, value=10, id='te-slider', tooltip={"placement": "bottom", "always_visible": False}),
                html.Label("TI (ms)", style={'color': '#aaa', 'fontSize': '14px'}),
                dcc.Slider(1, 5000, 10, value=400, id='ti-slider', tooltip={"placement": "bottom", "always_visible": False}),
                html.Label("Flip Angle (deg)", style={'color': '#aaa', 'fontSize': '14px'}),
                dcc.Slider(1, 180, 1, value=90, id='fa-slider', tooltip={"placement": "bottom", "always_visible": False}),
                html.Label("b-value (s/mm²)", style={'color': '#aaa', 'fontSize': '14px'}),
                dcc.Slider(0, 5000, 50, value=1000, id='bval-slider', tooltip={"placement": "bottom", "always_visible": False}),
            ], id='sidebar', style={'width': '25%', 'padding': '20px', 'backgroundColor': '#2c2c2c', 'height': '100vh', 'overflowY': 'auto', 'boxSizing': 'border-box', 'boxShadow': '2px 0 5px rgba(0,0,0,0.5)', 'zIndex': '10'}),
            
            # 3-Plane Views
            html.Div([
                html.Div([
                    dcc.Graph(id='sagittal-graph', config={'displayModeBar': False}, style={'height': '50vh', 'width': '100%'}),
                    dcc.Graph(id='coronal-graph', config={'displayModeBar': False}, style={'height': '50vh', 'width': '100%'}),
                ], style={'width': '50%', 'display': 'flex', 'flexDirection': 'column'}),
                html.Div([
                    dcc.Graph(id='axial-graph', config={'displayModeBar': False}, style={'height': '100vh', 'width': '100%'}),
                ], style={'width': '50%', 'display': 'flex', 'flexDirection': 'column'})
            ], style={'width': '75%', 'display': 'flex', 'flexDirection': 'row', 'backgroundColor': '#111'})
        ], style={'display': 'flex', 'flexDirection': 'row', 'height': '100vh', 'width': '100vw'})
    ], style={'backgroundColor': '#111', 'margin': '0', 'padding': '0', 'fontFamily': 'sans-serif', 'overflow': 'hidden'})

    def parse_contents(contents, filename):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        # Write to temp file to read with nibabel
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz' if filename.endswith('.gz') else '.nii') as temp_file:
            temp_file.write(decoded)
            temp_path = temp_file.name
            
        data = _load_nifti(Path(temp_path))
        os.unlink(temp_path)
        return data

    @app.callback(
        [Output('data-refresh-trigger', 'children'),
         Output('sequence-dropdown', 'options'),
         Output('sequence-dropdown', 'value'),
         Output('slice-x-slider', 'max'), Output('slice-x-slider', 'value'),
         Output('slice-y-slider', 'max'), Output('slice-y-slider', 'value'),
         Output('slice-z-slider', 'max'), Output('slice-z-slider', 'value')],
        [Input('upload-generic', 'contents'),
         Input('upload-t1', 'contents'),
         Input('upload-t2', 'contents'),
         Input('upload-m0', 'contents'),
         Input('upload-adc', 'contents')],
        [State('upload-generic', 'filename'),
         State('upload-t1', 'filename'),
         State('upload-t2', 'filename'),
         State('upload-m0', 'filename'),
         State('upload-adc', 'filename'),
         State('sequence-dropdown', 'value'),
         State('slice-x-slider', 'max'), State('slice-x-slider', 'value'),
         State('slice-y-slider', 'max'), State('slice-y-slider', 'value'),
         State('slice-z-slider', 'max'), State('slice-z-slider', 'value')]
    )
    def handle_uploads(gen_c, t1_c, t2_c, m0_c, adc_c, gen_f, t1_f, t2_f, m0_f, adc_f, curr_seq, xmax, xval, ymax, yval, zmax, zval):
        ctx_trig = ctx.triggered_id
        
        changed = False
        if ctx_trig == 'upload-generic' and gen_c:
            global_store['base'] = parse_contents(gen_c, gen_f)
            changed = True
        elif ctx_trig == 'upload-t1' and t1_c:
            global_store['t1'] = parse_contents(t1_c, t1_f)
            changed = True
        elif ctx_trig == 'upload-t2' and t2_c:
            global_store['t2'] = parse_contents(t2_c, t2_f)
            changed = True
        elif ctx_trig == 'upload-m0' and m0_c:
            global_store['m0'] = parse_contents(m0_c, m0_f)
            changed = True
        elif ctx_trig == 'upload-adc' and adc_c:
            global_store['adc'] = parse_contents(adc_c, adc_f)
            changed = True
            
        if changed or ctx_trig is None: # Initial load
            update_ref_shape()
            
            if global_store['m0'] is None and (global_store['t1'] is not None or global_store['t2'] is not None):
                 global_store['m0'] = np.ones_like(global_store['ref']) * 1000.0
                 
            s = global_store['shape']
            nx, ny, nz = s[0]-1, s[1]-1, s[2]-1
            
            # Try to keep values proportional if max changed by a lot, else keep them
            if xmax != nx: xval = nx // 2
            if ymax != ny: yval = ny // 2
            if zmax != nz: zval = nz // 2
            
            choices = get_seq_choices()
            opts = [{'label': s, 'value': s} for s in choices]
            val = curr_seq if curr_seq in choices else choices[0]
            
            return "Updated", opts, val, nx, xval, ny, yval, nz, zval
            
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    @app.callback(
        [Output('sagittal-graph', 'figure'),
         Output('coronal-graph', 'figure'),
         Output('axial-graph', 'figure')],
        [Input('data-refresh-trigger', 'children'),
         Input('sequence-dropdown', 'value'),
         Input('slice-x-slider', 'value'),
         Input('slice-y-slider', 'value'),
         Input('slice-z-slider', 'value'),
         Input('tr-slider', 'value'),
         Input('te-slider', 'value'),
         Input('ti-slider', 'value'),
         Input('fa-slider', 'value'),
         Input('bval-slider', 'value')]
    )
    def update_graphs(trigger, sequence, x, y, z, tr, te, ti, fa, bval):
        
        t1_map = global_store['t1']
        t2_map = global_store['t2']
        m0_map = global_store['m0']
        adc_map = global_store['adc']
        base_img = global_store['base']
        ref_map = global_store['ref']
        shape = global_store['shape']

        def get_slice(plane, idx):
            def slice_array(arr):
                if arr is None: return None
                # Generate standard neuroimaging display conventions (rot90)
                if plane == 0: 
                    return np.rot90(arr[idx, :, :])
                elif plane == 1: 
                    return np.rot90(arr[:, idx, :])
                elif plane == 2: 
                    return np.rot90(arr[:, :, idx])

            t1_s = slice_array(t1_map)
            t2_s = slice_array(t2_map)
            m0_s = slice_array(m0_map)
            adc_s = slice_array(adc_map)
            ref_s = slice_array(ref_map)
            base_s = slice_array(base_img)

            if sequence == "Generic":
                return base_s if base_s is not None else (t1_s if t1_s is not None else ref_s)
            
            t1 = t1_s if t1_s is not None else np.zeros_like(ref_s)
            t2 = t2_s if t2_s is not None else np.zeros_like(ref_s)
            m0 = m0_s if m0_s is not None else np.zeros_like(ref_s)
            
            if sequence == "SE":
                return synthetic_se(t1, t2, m0, tr, te)
            elif sequence == "SPGR (FLASH)":
                return synthetic_spgr(t1, m0, tr, fa)
            elif sequence == "IR":
                return synthetic_ir(t1, t2, m0, tr, te, ti)
            elif sequence == "FLAIR":
                return synthetic_flair(t1, t2, m0, tr, te, ti)
            elif sequence == "bSSFP":
                return synthetic_ssfp(t1, t2, m0, tr, fa)
            elif sequence == "MPRAGE":
                return synthetic_mprage(t1, m0, tr, ti, fa)
            elif sequence == "DWI":
                if adc_s is not None:
                    s0 = m0 if m0_s is not None else np.ones_like(adc_s) * 1000.0
                    return synthetic_dwi(adc_s, s0, bval)
                else:
                    return np.zeros_like(ref_s)
            return ref_s

        sag_slice = get_slice(0, x)
        cor_slice = get_slice(1, y)
        ax_slice = get_slice(2, z)

        def make_fig(data_slice, title, h_line=None, v_line=None):
            if data_slice is None:
                data_slice = np.zeros((10, 10))
            
            # Simple global contrast min/max
            vmin = np.percentile(data_slice, 1)
            vmax = np.percentile(data_slice, 99)
            
            fig = go.Figure(data=go.Heatmap(
                z=data_slice,
                colorscale='gray',
                showscale=False,
                hoverinfo='none',
                zmin=vmin,
                zmax=vmax
            ))
            fig.update_layout(
                title=dict(text=title, font=dict(color='white', size=16), pad=dict(t=10, b=10)),
                margin=dict(l=5, r=5, t=40, b=5),
                paper_bgcolor='#111',
                plot_bgcolor='#111',
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
            )
            
            # Crosshairs overlay
            if h_line is not None:
                # The Y axis on a heatmap with np.rot90 corresponds to the shape after rotation.
                # Adding an explicit line over the exact matrix index is a bit tricky with rot90
                # We will draw a green line through the center of the crosshair.
                fig.add_hline(y=data_slice.shape[0]-h_line-1, line_color="rgba(0, 255, 0, 0.4)", line_width=1)
            if v_line is not None:
                fig.add_vline(x=v_line, line_color="rgba(0, 255, 0, 0.4)", line_width=1)
                
            return fig

        # Map the slicer coordinates to the 2D rot90 view coordinates
        fig_sag = make_fig(sag_slice, "Sagittal", h_line=z, v_line=y)
        fig_cor = make_fig(cor_slice, "Coronal", h_line=z, v_line=x)
        
        h_line_ax = shape[1]-y-1 if len(shape) > 1 else y
        fig_ax = make_fig(ax_slice, "Axial", h_line=h_line_ax, v_line=x)

        return fig_sag, fig_cor, fig_ax

    def find_free_port(start_port=8050, max_port=8100):
        for p in range(start_port, max_port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', p)) != 0:
                    return p
        return start_port

    active_port = find_free_port(port)

    # Run dash in a background thread
    def run_dash():
        # turn off debug/reloader
        app.run(port=active_port, debug=False, use_reloader=False)

    t = threading.Thread(target=run_dash, daemon=True)
    t.start()
    
    # Wait up to 10 seconds for the Dash server to initialize and bind to the port
    timeout = 10.0
    start_time = time.time()
    server_ready = False
    while time.time() - start_time < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', active_port)) == 0:
                server_ready = True
                break
        time.sleep(0.1)
        
    if not server_ready:
        print(f"Warning: Dash server did not seem to bind to port {active_port} in time.", file=sys.stderr)
    else:
        # Give it a tiny buffer after port binds before hitting it
        time.sleep(0.2)
    
    # Launch pywebview natively
    webview.create_window(
        "qMRI Neuropipe Viewer",
        f"http://127.0.0.1:{active_port}",
        width=1400,
        height=900,
        background_color="#111111"
    )
    webview.start()
