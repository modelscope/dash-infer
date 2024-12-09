'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    conf.py
'''
# Configuration file for the Sphinx documentation builder.
import recommonmark
from recommonmark.transform import AutoStructify

# -- Project information

project = 'DashInfer'
copyright = '2024, Alibaba.inc'
author = 'DashInfer Team'

release = '2.0.0'
version = '2.0.0'

# -- General configuration

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

language = 'en'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = 'sphinx'


extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'recommonmark',
    'sphinx_markdown_tables'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
#    'dashinfer' : ('https://github.com/modelscope/dash-infer/', None)
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
html_static_path = ['_static']
html_logo = "_static/logo.png"

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
#html_theme = 'agogo'
#html_theme_options = {
#    "pagewidth": "120em",
#    "documentwidth":"100em",
#}

# -- Options for EPUB output
epub_show_urls = 'footnote'

def setup(app):
    app.add_css_file('my_theme.css')
    app.add_config_value('recommonmark_config', {
            # 'url_resolver': lambda url: github_doc_root + url,
            'auto_toc_tree_section': 'Contents',
            'enable_eval_rst': True,
            }, True)
    app.add_transform(AutoStructify)
