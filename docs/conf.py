# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

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
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
