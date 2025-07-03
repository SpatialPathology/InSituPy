# Configuration file for the Sphinx documentation builder.

import os
import sys
from pathlib import PurePosixPath

sys.path.insert(0, os.path.abspath('../..'))


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


# -- Project information

project = 'InSituPy'
copyright = '2025, Johannes Wirth'
author = 'Johannes Wirth'
release = get_version("../../insitupy/__init__.py")
version = get_version("../../insitupy/__init__.py")

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "sphinx_design", # can be used for things such as cards
    "myst_nb"
]



autosummary_generate = True
autodoc_process_signature = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
myst_heading_anchors = 3  # create anchors for h1-h3

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]

myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
    ".md": "myst-nb"
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

#templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_book_theme'
html_static_path = ["_static"]
html_title = project
html_logo = "_static/img/insitupy_logo_with_name_wo_bg.png"

html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

html_theme_options = {
    "repository_url": "https://github.com/SpatialPathology/InSituPy",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "path_to_docs": "./docs",
    "display_version": True,
    "logo_only": True,
    "version_selector": True,
    "collapse_navigation": False
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
