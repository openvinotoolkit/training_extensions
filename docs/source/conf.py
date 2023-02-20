# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'OpenVINO Training Extensions'
copyright = '2023, OpenVINO Training Extensions Contributors'
author = 'OpenVINO Training Extensions Contributors'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx_copybutton']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'spark_sphinx_theme'

#html_theme_options = {
#"navbar_start": ["navbar-logo"],
#"navbar_center": [],
#"navbar_end": ["navbar-icon-links"]
#}

html_static_path = ['_static']
html_theme_options = {
   "navbar_center": [],
   "logo": {
      "image_light": "_static/logos/otx-logo-black-mini.png",
      "image_dark": "_static/logos/otx-logo-black-mini.png",
   }
}

html_css_files = [
    '_static/css/custom.css',
]
