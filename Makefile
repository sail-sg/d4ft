SHELL          = /bin/bash
PROJECT_NAME   = d4ft
PROJECT_FOLDER = d4ft
PYTHON_FILES   = $(shell find . -type f -name "*.py" -not -path '*/.venv/*')
BAZEL_FILES    = $(shell find . -type f -name "*BUILD" -o -name "*.bzl" -not -path '*/.venv/*')
COMMIT_HASH    = $(shell git log -1 --format=%h)
COPYRIGHT      = "Garena Online Private Limited"
BAZELOPT       =
PATH           := $(HOME)/go/bin:$(PATH)

# installation
check_install = python3 -c "import $(1)" || (cd && pip3 install $(1) --upgrade && cd -)
check_install_extra = python3 -c "import $(1)" || (cd && pip3 install $(2) --upgrade && cd -)

flake8-install:
	$(call check_install, flake8)
	$(call check_install_extra, bugbear, flake8_bugbear)

py-format-install:
	$(call check_install, yapf)

cpplint-install:
	$(call check_install, cpplint)

# requires go >= 1.16
bazel-install:
	command -v bazel || (go install github.com/bazelbuild/bazelisk@latest && ln -sf $(HOME)/go/bin/bazelisk $(HOME)/go/bin/bazel)

buildifier-install:
	command -v buildifier || go install github.com/bazelbuild/buildtools/buildifier@latest

addlicense-install:
	command -v addlicense || go install github.com/google/addlicense@latest

doc-install:
	$(call check_install, pydocstyle)
	$(call check_install_extra, doc8, "doc8<1")
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	$(call check_install_extra, sphinxcontrib.spelling, sphinxcontrib.spelling pyenchant)

# python linter

flake8: flake8-install
	flake8 $(PYTHON_FILES) --count --show-source --statistics --exclude d4ft/integral/obara_saika/boys_table.py

flake8-fix: flake8-install
	flake8 $(PYTHON_FILES) --exclude d4ft/integral/obara_saika/boys_table.py

py-format: py-format-install
	yapf -dr $(PYTHON_FILES) --exclude d4ft/integral/obara_saika/boys_table.py

py-format-fix: py-format-install
	yapf -ir $(PYTHON_FILES) --exclude d4ft/integral/obara_saika/boys_table.py

# bazel file linter

buildifier: buildifier-install
	buildifier -r -lint=warn $(BAZEL_FILES)

buildifier-fix: buildifier-install
	buildifier -r -lint=fix $(BAZEL_FILES)

# documentation

addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) -l apache -check $(PROJECT_FOLDER)

addlicense-fix: addlicense-install
	addlicense -c $(COPYRIGHT) -l apache $(PROJECT_FOLDER)

docstyle: doc-install
	pydocstyle $(PROJECT_NAME) && doc8 docs && cd docs && make html SPHINXOPTS="-W"

doc: doc-install
	cd docs && make html && cd _build/html && python3 -m http.server

doc-clean:
	cd docs && make clean

lint: buildifier flake8 py-format # docstyle

format: py-format-install buildifier-install addlicense-install py-format-fix buildifier-fix addlicense-fix

bazel-test: bazel-install
	bazel test --test_output=all //tests/... --config=test --spawn_strategy=local --color=yes
