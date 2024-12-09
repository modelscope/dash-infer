# DashInfer Documents

The document source codes build by Sphinx, and most documents are in reStructuredText format. 


## Install Sphinx

```
pip install sphinx sphinx_markdown_tables sphinx-rtd-theme recommonmark
```


## Build


```
cd docs
make html
```

## Preview

- macOS terminal: open docs/build/html/index.html

- vscode:
1. Install extension `Live Server`.
2. Open file `docs/build/html/index.html`.
3. Right click and choose `Open with Live Server`.
