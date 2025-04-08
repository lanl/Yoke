# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Yoke'
copyright = '2025, Los Alamos National Labs. Produced by: Kyle Hickmann, Skylar Callis, Gal Egozi, Soumi De, Bryan Kaiser, Sourabh Pandit, Sharmistha Chakrabarti, Derek Armstrong, Andrew Henrick, David Schodt'
author = 'Kyle Hickmann, Skylar Callis, Gal Egozi, Soumi De, Bryan Kaiser, Sourabh Pandit, Sharmistha Chakrabarti, Derek Armstrong, Andrew Henrick, David Schodt'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Optional: for Google style docstrings
    'sphinx.ext.viewcode',  # Optional: to add links to source code
    'sphinx.ext.coverage',  # Optional: for GitHub Pages deployment
    'sphinx.ext.mathjax',   # Render math in docstrings.
]

# Napolean settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# Other settings
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_title = "Get Yoked!"
#html_logo = "path/to/logo.png"
#html_static_path = ['_static']

html_theme_options = {
    "sidebar_hide_name": False,  # optional
    "navigation_with_keys": True,  # keyboard nav
}

html_sidebars = {
    "**": ["sidebar/brand.html", "sidebar/navigation.html"],  # remove toc.html
}

