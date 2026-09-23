# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys, os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 (documentation CI)
    import tomli as tomllib

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'regdiffusion'
copyright = '2025, Hao Zhu, Donna Slonim'
author = 'Hao Zhu, Donna Slonim'
with (Path(__file__).resolve().parents[1] / 'pyproject.toml').open('rb') as f:
    package_metadata = tomllib.load(f)['project']
release = package_metadata['version']

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    # "sphinx_panels",
    'myst_parser',
    'sphinx_sitemap',
]
html_extra_path = ['supplements']

copybutton_prompt_text = ">>> "

autosummary_generate = True
numpydoc_show_class_members = False

source_suffix = ['.rst', '.md']
templates_path = ['_templates']
exclude_patterns = ['_build', 'BUILD.md', 'Thumbs.db', '.DS_Store', '.ipynb_checkpoints', '__pycache__/']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_logo = "_static/rd_logo_horizontal.png"
html_title = 'RegDiffusion Documentation'
html_baseurl = 'https://tuftsbcb.github.io/RegDiffusion/'
language = 'en'
sitemap_url_scheme = '{link}'
sitemap_locales = [None]
sitemap_excludes = ['search.html', 'genindex.html', 'py-modindex.html', '_modules/*']

html_context = {
    'page_descriptions': {
        'index': 'Infer gene regulatory networks from single-cell RNA-seq with RegDiffusion, an open-source Python package using diffusion models. Tutorials, API, and paper.',
        'quick_tour': 'Learn how to infer, evaluate, export, and visualize a gene regulatory network from single-cell expression data using the RegDiffusion Python API.',
        'visualizing_grn': 'Visualize RegDiffusion gene regulatory networks with lightgraph using a human PBMC example and Reactome pathway annotations.',
        'large_networks': 'Scale RegDiffusion to large gene regulatory networks with memory-efficient training, sparse expression matrices, and GPU memory benchmarks.',
        'downstream_with_pyscenic': 'Use RegDiffusion-inferred gene regulatory networks for downstream pySCENIC analysis of single-cell RNA-seq data.',
        'faq': 'RegDiffusion questions answered: input data, sparse matrices, GPU and CPU support, pySCENIC integration, benchmarks, and citation.',
        'main_api': 'RegDiffusion Python API reference for network training, GRN storage and visualization, and evaluation against reference networks.',
        'models': 'Python reference for the diffusion models used by RegDiffusion to infer gene regulatory networks.',
        'data_module': 'Load BEELINE benchmarks and preprocessed mouse microglia expression datasets for RegDiffusion network inference and evaluation.',
        'supplements': 'Explore supplementary local gene regulatory networks from the RegDiffusion research paper.',
    },
    'software_schema': {
        '@context': 'https://schema.org',
        '@type': 'SoftwareSourceCode',
        'name': 'RegDiffusion',
        'description': package_metadata['description'],
        'url': html_baseurl,
        'codeRepository': 'https://github.com/TuftsBCB/RegDiffusion',
        'downloadUrl': 'https://pypi.org/project/regdiffusion/',
        'programmingLanguage': 'Python',
        'license': 'https://www.apache.org/licenses/LICENSE-2.0',
        'version': release,
        'author': [
            {'@type': 'Person', 'name': 'Hao Zhu'},
            {'@type': 'Person', 'name': 'Donna Slonim'},
        ],
        'citation': {
            '@type': 'ScholarlyArticle',
            'name': 'From Noise to Knowledge: Diffusion Probabilistic Model-Based Neural Inference of Gene Regulatory Networks',
            'identifier': '10.1089/cmb.2024.0607',
            'url': 'https://doi.org/10.1089/cmb.2024.0607',
        },
    },
}

html_theme_options = {
    "repository_url": "https://github.com/TuftsBCB/RegDiffusion",
    "use_repository_button": True,
}
