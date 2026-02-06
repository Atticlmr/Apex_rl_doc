# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ApexRL'
copyright = '2026, Atticlmr'
author = 'Atticlmr'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


html_context = {
    'display_github': True,
    'github_user': 'yourname',
    'github_repo': 'yourproject',
    'github_version': 'main',
    'conf_py_path': '/docs/',
    

    'current_version': 'latest',
    'version': '0.0.1',
    'versions': [
        ('latest', '/en/latest/'),
        ('stable', '/en/stable/'),
        ('1.0', '/en/2.0/'),
        ('1.0', '/en/1.0/'),
    ],
}

html_theme_options = {
    'display_version': True,
    'version_selector': True,
}
