# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "OpenVINO Training Extensions"
copyright = "2022, OpenVINO Training Extensions Contributors"
author = "OpenVINO Training Extensions Contributors"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinxemoji.sphinxemoji",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
    "autoapi.extension",
    "sphinx_rtd_theme",
]

autoapi_dirs = ["../../otx"]
autoapi_root = "api"
autoapi_type = "python"

autosummary_generate = True
autodoc_member_order = "groupwise"
autoclass_content = "both"

autodoc_default_options = {
    "members": True,
    "methods": True,
    "special-members": "__call__",
    "exclude-members": "_abc_impl",
    "show-inheritance": True,
}

myst_enable_extensions = ["colon_fence"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2
