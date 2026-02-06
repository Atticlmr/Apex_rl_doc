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

import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

extensions = [
    'sphinx.ext.autodoc',           # 核心：从 docstring 生成文档
    'sphinx.ext.autosummary',       # 生成摘要表格
    'sphinx.ext.napoleon',          # 支持 Google/NumPy docstring 风格
    'sphinx.ext.viewcode',          # 添加源码链接
    'sphinx_autodoc_typehints',     # 自动提取类型注解
]

# 自动生成 autosummary 文件
autosummary_generate = True

# 类型提示显示在描述中
autodoc_typehints = "description"

# 默认 autodoc 选项
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
