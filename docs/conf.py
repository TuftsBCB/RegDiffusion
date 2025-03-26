# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys, os
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'regdiffusion'
copyright = '2025, Hao Zhu, Donna Slonim'
author = 'Hao Zhu, Donna Slonim'
release = '0.0.9'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    # "sphinx_panels",
    'myst_parser'
]
html_extra_path = ['supplements']

copybutton_prompt_text = ">>> "

autosummary_generate = True
numpydoc_show_class_members = False

source_suffix = ['.rst', '.md']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.ipynb_checkpoints', '__pycache__/']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_logo = "_static/rd_logo_horizontal.png"

html_theme_options = {
    "repository_url": "https://github.com/TuftsBCB/RegDiffusion",
    "use_repository_button": True,
}
