# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information

project = 'D4FT'
copyright = '2022, SEA AI LAB'

# release = '0.0'
version = '0.0.0'

# -- General configuration
myst_enable_extensions = ["amsmath", "dollarmath"]
extensions = [
  'myst_parser',
  'sphinx.ext.duration',
  'sphinx.ext.doctest',
  'sphinx.ext.napoleon',
  'sphinx.ext.autodoc',
  'sphinx.ext.autosummary',
  'sphinx.ext.intersphinx',
  'sphinx.ext.mathjax',
]

intersphinx_mapping = {
  'python': ('https://docs.python.org/3/', None),
  'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_book_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- customize theme color

html_theme_options = {
  # 'style_nav_header_background': '#910f0f',
  'show_toc_level': 2,
}

# include the physics package
# https://stackoverflow.com/questions/75459170
mathjax3_config = {
  'loader': {
    'load': ['[tex]/physics']
  },
  'tex': {
    'packages': {
      '[+]': ['physics']
    }
  },
}

# -- Options for reading type annotation
napoleon_google_docstring = True
napoleon_numpy_docstring = False  # set this to True if you also use numpy-style docstrings
napoleon_include_init_with_doc = True
