from setuptools import setup, find_packages


setup_requirements = ["pytest-runner", "setuptools_scm"]


def requirements(fp: str):
    with open(fp) as f:
        return [r.strip() for r in f.readlines()]


setup(
    author="Miles Granger",
    author_email="milg@equinor.no",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Unlicense",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="Train and build models for Argo / Kubernetes",
    entry_points={"console_scripts": ["gordo-components=gordo_components.cli:gordo"]},
    install_requires=requirements("requirements.in"),  # Allow flexible deps for install
    license="Unlicense",
    name="gordo-components",
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=requirements("test_requirements.txt"),
    url="https://github.com/Statoil/gordo-flow",
    use_scm_version={
        "write_to": "gordo_components/_version.py",
        "relative_to": __file__,
    },
    zip_safe=True,
    package_data={
        "": [
            "gordo_components/workflow/workflow_generator/resources/argo-workflow.yml.template"
        ]
    },
    include_package_data=True,
)
