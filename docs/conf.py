# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import datetime
import importlib
import inspect

_module_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _module_path)
_examples_path = os.path.join(os.path.dirname(__file__), "..", "examples")
# sys.path.insert(0, _examples_path)

import gordo

from gordo.util.version import parse_version, GordoRelease

project = "gordo"
copyright = f"2019-{datetime.date.today().year}, Equinor"
author = "Equinor ASA"
version = gordo.__version__
_parsed_version = parse_version(version)
commit = f"{version}" if type(_parsed_version) is GordoRelease and not _parsed_version.suffix else "HEAD"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx.ext.linkcode",
    "sphinx.ext.autosectionlabel",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_copybutton",
    "sphinx_click",
    "nbsphinx"
]

root_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = [".rst", ".md"]

code_url = f"https://github.com/equinor/{project}/blob/{commit}"

_ignore_linkcode_infos = [
    # caused "OSError: could not find class definition"
    {"module": "gordo_core.utils", "fullname": "PredictionResult"},
    {'module': 'gordo.workflow.config_elements.schemas', 'fullname': 'Model.Config.extra'},
    {'module': 'gordo.reporters.postgres', 'fullname': 'Machine.DoesNotExist'}
]


def linkcode_resolve(domain, info):
    if domain != "py":
        return None

    for ignore_info in _ignore_linkcode_infos:
        if (
            info["module"] == ignore_info["module"]
            and info["fullname"] == ignore_info["fullname"]
        ):
            return None

    mod = importlib.import_module(info["module"])
    if "." in info["fullname"]:
        objname, attr = info["fullname"].split(".")
        obj = getattr(mod, objname)
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            return None
    else:
        obj = getattr(mod, info["fullname"])

    try:
        file = inspect.getsourcefile(obj)
        lines = inspect.getsourcelines(obj)
    except TypeError:
        return None

    rel_path = os.path.relpath(file, os.path.abspath(".."))
    if not rel_path.startswith("gordo"):
        return None
    start, end = lines[1], lines[1] + len(lines[0]) - 1
    return f"{code_url}/{rel_path}#L{start}-L{end}"


# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

extlinks = {
    "issue": ("https://github.com/equinor/gordo/issues/%s", "Issue #"),
    "pr": ("https://github.com/equinor/gordo/pull/%s", "PR #"),
    "user": ("https://github.com/%s", "@"),
}

intersphinx_mapping = {
    "gordo-core": ("https://gordo-core.readthedocs.io/en/latest/", None),
    "gordo-client": ("https://gordo-client.readthedocs.io/en/latest/", None),
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "numexpr": ("https://numexpr.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    # "pyarrow": ("https://arrow.apache.org/docs/python/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "influxdb": ("https://influxdb-python.readthedocs.io/en/latest/", None),
    "flask": ("https://flask.palletsprojects.com/", None),
}

autosectionlabel_prefix_document = True

autodoc_typehints = "signature"

autodoc_typehints_description_target = "documented"

autodoc_mock_imports = ["tensorflow"]

# Document both class doc (default) and documentation in __init__
autoclass_content = "both"

# Use docstrings from parent classes if not exists in children
autodoc_inherit_docstrings = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/equinor/gordo",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "source_repository": "https://github.com/equinor/gordo",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_copy_source = False

html_show_sphinx = False

# Configs for different output formats
# ------------------------------------

latex_elements = {
    "pointsize": "12pt",
}

latex_documents = [
    (
        root_doc,
        "Gordo.tex",
        "Gordo Documentation",
        "Equinor ASA",
        "manual",
    )
]

man_pages = [(root_doc, "gordo", "Gordo Documentation", [author], 1)]

epub_title = project
epub_exclude_files = ["search.html"]
