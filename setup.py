from setuptools import setup, find_packages


setup_requirements = ["pytest-runner", "setuptools_scm"]


def requirements(fp: str):
    with open(fp) as f:
        return [r.strip() for r in f.readlines()]


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
    install_requires=requirements("requirements.in"),  # Allow flexible deps for install
    license="AGPLv3",
    name="gordo",
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=requirements("test_requirements.txt"),
    url="https://github.com/equinor/gordo",
    use_scm_version={
        "write_to": "gordo/_version.py",
        "relative_to": __file__,
    },
    zip_safe=True,
    package_data={
        "": [
            "python/gordo/workflow/workflow_generator/resources/argo-workflow.yml.template"
        ]
    },
    include_package_data=True,
)
