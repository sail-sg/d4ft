# AI for science 

## Install

```bash
pip install -e .
```

## Tutorial and Documentation for Quantum Chemistry

### Viewing in the Browser

``` shell
cd docs
pip install -r requirements.txt  # install the tools needs to build the website
make html  # generate a static site from the rst and markdown files
sphinx-serve  # run a server locally, so that it can be viewed in browser
sphinx-autobuild docs docs/_build/html
```


#### Auto-build (optional)

```shell
pip install sphinx-autobuild
sphinx-autobuild docs docs/_build/html
```

### Editing

The `conf.py` has been setup to support markdown, so we can mix `rst` and `md` files in this project. For example, a `test.md` file is created at `docs/math/test.md`, and it is added to `docs/math/index.rst`.

