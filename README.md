# ApexRL Documentation

This directory contains the documentation for ApexRL, built with Sphinx and the Furo theme.

## Structure

```
docs/
├── source/           # English documentation
│   ├── index.rst
│   ├── tutorials/
│   ├── modules/
│   └── API/
├── source_zh/        # Chinese documentation
│   ├── index.rst
│   ├── tutorials/
│   ├── modules/
│   └── API/
├── requirements.txt  # Documentation dependencies
└── .readthedocs.yaml # ReadTheDocs configuration
```

## Local Build

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Build English Documentation

```bash
cd docs
make html
```

The built documentation will be in `build/html/`.

### Build Chinese Documentation

```bash
cd docs
sphinx-build -b html source_zh build/html_zh
```

## Serving Locally

```bash
cd docs/build/html
python -m http.server 8000
```

Then open http://localhost:8000 in your browser.

## Adding Content

### English
Add new `.rst` files to `source/` and update `source/index.rst` to include them in the toctree.

### Chinese
Add new `.rst` files to `source_zh/` and update `source_zh/index.rst` accordingly.

## Style Guide

- Use `autoclass` and `autofunction` for API documentation
- Include code examples in tutorials
- Keep line length under 100 characters for readability
- Use cross-references with `:doc:` and `:ref:`

## ReadTheDocs Configuration

The documentation is automatically built and hosted on ReadTheDocs:
- English: https://apex-rl-doc.readthedocs.io/en/latest/
- Chinese: https://apex-rl-doc.readthedocs.io/zh/latest/

## Translation Workflow

When adding new content:
1. Add to English documentation first
2. Create corresponding Chinese version
3. Keep both versions in sync
