# Configuration file for the Sphinx documentation builder.
import os
import sys
# Point to the source code (assuming this directory is moved to repo/docs)
sys.path.insert(0, os.path.abspath('../src'))

# Mock heavy dependencies to avoid installation on RTD
autodoc_mock_imports = [
    'ants', 
    'antspyx',
    'dipy', 
    'dmipy', 
    'nibabel', 
    'numpy', 
    'scipy', 
    'pandas', 
    'matplotlib', 
    'rich', 
    'typer', 
    'pydantic',
    'yaml',
    'jinja2' 
]

# -- Project information -----------------------------------------------------
project = 'qmri-neuropipe'
copyright = '2025, Developing Brain Imaging Lab'
author = 'Developing Brain Imaging Lab'
release = '2.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',    # Support Google-style docstrings
    'sphinx.ext.viewcode',    # Add links to highlighted source code
    'sphinx.ext.todo',
    'myst_parser',            # Support Markdown files
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
