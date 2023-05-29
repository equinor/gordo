# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -W
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

_static/architecture_diagram.png:
	python3 ./_static/architecture_diagram.py

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile _static/architecture_diagram.png
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

watch: _static/architecture_diagram.png
	sphinx-autobuild --watch "$(SOURCEDIR)/../gordo" "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
