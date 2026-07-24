# Configuration file for the Sphinx documentation builder.

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version

sys.path.insert(0, os.path.abspath('../..'))

try:
    release = _get_version("insitupy-spatial")
except PackageNotFoundError:
    release = "dev"
version = ".".join(release.split(".")[:2])

# -- Project information

project = 'InSituPy'
copyright = '2025, Johannes Wirth'
author = 'Johannes Wirth'

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
    "myst_nb",
    "sphinxcontrib.mermaid",
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

# Render GitHub-flavored ```mermaid fences as mermaid directives. Without this, MyST treats them
# as plain code blocks and the diagrams silently degrade to grey text boxes here while still
# rendering fine on GitHub - so one fence can be the single source for both surfaces.
myst_fence_as_directive = ["mermaid"]

# Pin the mermaid runtime. sphinxcontrib-mermaid loads it from a CDN at page-view time and
# defaults to whatever version the installed extension ships with; since docs/requirements.txt
# pins nothing, that would let the renderer change under us between builds. Bump deliberately.
mermaid_version = "11.12.1"
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
    "path_to_docs": "./docs/source",
    # "display_version": True,
    # "logo_only": True,
    # "version_selector": True,
    "collapse_navigation": False
}

html_context = {
    "display_github": True,
    "github_user": "SpatialPathology",
    "github_repo": "InSituPy",
    #"github_version": os.environ.get("READTHEDOCS_VERSION", "main"),
    "github_version": "dev_reader",
    "conf_py_path": "/docs/source/",
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
