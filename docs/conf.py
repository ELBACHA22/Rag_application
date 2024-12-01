# conf.py

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(".."))  # Adjust path if needed

# -- Project information -----------------------------------------------------
project = "RAG PDF Chatbot"
author = "EL BACHA Ikram"
release = "1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",       # Automatically document your code
    "sphinx.ext.napoleon",      # Support for NumPy and Google style docstrings
    "sphinx.ext.viewcode",      # Link to source code
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
