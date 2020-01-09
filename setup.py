import os
from setuptools import setup, find_packages


setup_requirements = ["pytest-runner", "setuptools_scm"]


def requirements(*fps):
    reqs = []
    for fp in fps:
        with open(os.path.join("requirements", fp)) as f:
            reqs.extend(
                [
                    r.strip()
                    for r in f.readlines()
                    if ("==" in r or "~=" in r) and (not r.startswith("#"))
                ]
            )
    return reqs


setup(
    author="Equinor ASA",
    author_email="fg_gpl@equinor.com",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="Train and build models for Argo / Kubernetes",
    entry_points={"console_scripts": ["gordo=gordo.cli:gordo"]},
    install_requires=["gordo[full]"],
    license="AGPLv3",
    name="gordo",
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=["gordo[tests]"],
    url="https://github.com/equinor/gordo",
    use_scm_version={"write_to": "gordo/_version.py", "relative_to": __file__},
    zip_safe=True,
    package_data={
        "": [
            "python/gordo/workflow/workflow_generator/resources/argo-workflow.yml.template"
        ]
    },
    extras_require={
        "docs": requirements("docs_requirements.in", "core_requirements.in"),
        "core": requirements("core_requirements.in"),
        "full": requirements("core_requirements.in", "full_requirements.in"),
        "test": requirements(
            "core_requirements.txt", "test_requirements.txt", "full_requirements.txt"
        ),
    },
    include_package_data=True,
)
