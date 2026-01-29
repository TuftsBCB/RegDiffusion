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
pip install sphinx sphinx-book-theme sphinx-copybutton myst-parser sphinxext-napoleon
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
