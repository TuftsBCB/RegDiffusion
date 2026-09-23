# Building Sphinx Documentation

## Quick Start

To rebuild the documentation, navigate to the `docs/` directory and run:

```bash
cd docs
make html
```

The built HTML documentation will be available in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser to view it.

## Common Commands

### Build HTML Documentation
```bash
make html
```

### Clean Build Directory
If you encounter issues, clean the build directory first:
```bash
make clean
make html
```

### View Available Build Targets
```bash
make help
```

### Other Build Formats
- `make latexpdf` - Build PDF documentation
- `make epub` - Build EPUB format
- `make linkcheck` - Check all external links

## Troubleshooting

### If `make` command is not found:
Use the Python module directly:
```bash
python -m sphinx -b html . _build/html
```

### If dependencies are missing:
Install required packages:
```bash
pip install -r requirements.txt
```

### If autodoc fails to import modules:
Make sure you're in the correct conda environment and the package is installed:
```bash
pip install -e ..
```

## File Structure

- `conf.py` - Sphinx configuration file
- `index.rst` - Main documentation entry point
- `*.rst` - ReStructuredText source files
- `*.md` - Markdown files (via myst-parser)
- `_build/` - Build output directory (generated)
- `_static/` - Static assets (images, CSS, etc.)
- `_templates/` - Custom templates

## Search and discovery

The HTML build generates page descriptions, Open Graph metadata, canonical URLs,
a `sitemap.xml`, and software structured data on the homepage. The documentation
version is read from `pyproject.toml`. Update `page_descriptions` in `conf.py` when
adding a landing page, and keep structured data consistent with visible content.

After deployment, submit `https://tuftsbcb.github.io/RegDiffusion/sitemap.xml` in
Google Search Console and Bing Webmaster Tools. Inspect the homepage and FAQ to
confirm that the deployed pages can be indexed. Track non-brand queries such as
"single-cell gene regulatory network inference" and referrals from AI search,
alongside branded RegDiffusion queries. Record dates and cited URLs when manually
checking AI answers; one answer is not a reliable ranking measurement.

This project is hosted under `/RegDiffusion/`. A robots.txt in this repository's
published directory would not control crawling: robots.txt must be served from
the origin root (`https://tuftsbcb.github.io/robots.txt`), managed by the organization
site. Review that root file if crawling is blocked. Special AI files or metadata
do not guarantee indexing or AI citations.
