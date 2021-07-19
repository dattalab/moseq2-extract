# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'moseq2-extract'
author = 'Datta Lab'


# -- General configuration ---------------------------------------------------

on_rtd = input('Generating a pdf? (y/[n])')
if on_rtd == 'y':
    on_rtd = False
else:
    on_rtd = True

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
if on_rtd:
    extensions = [
        'sphinx.ext.napoleon',
        'sphinx.ext.autodoc',
        'sphinx.ext.autosummary',
        'sphinx_click.ext'
    ]
else:
    extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'rst2pdf.pdfbuilder',
    'sphinx_click.ext'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True  # Make _autosummary files and include them
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

autodoc_default_flags = [
         # Make sure that any autodoc declarations show the right members
         "members",
         "inherited-members",
         "private-members",
         "show-inheritance",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']