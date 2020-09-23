import os
from setuptools import setup, find_packages


setup_requirements = ["pytest-runner", "setuptools_scm"]
on_rtd = os.environ.get("READTHEDOCS") == "True"


def requirements(fp: str):
    with open(os.path.join(os.path.dirname(__file__), "requirements", fp)) as f:
        return [
            r.strip()
            for r in f.readlines()
            if r.strip() and not r.startswith("#") and not r.startswith("-")
        ]


extras_require = {
    "docs": requirements("docs_requirements.in"),
    "mlflow": requirements("mlflow_requirements.in"),
    "postgres": requirements("postgres_requirements.in"),
    "tests": requirements("test_requirements.txt"),
}
extras_require["full"] = extras_require["mlflow"] + extras_require["postgres"]

install_requires = requirements("requirements.in")  # Allow flexible deps for install

# Read the docs have quite low memory limits on their build servers, so low
# that pip crashes when downloading tensorflow. So we must remove it from the install
# requirements, and set up autodoc_mock_imports in docs/conf.py
if on_rtd:
    install_requires = [req for req in install_requires if "tensorflow" not in req]

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
    install_requires=install_requires,
    license="AGPLv3",
    name="gordo",
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=extras_require["tests"],
    extras_require=extras_require,
    url="https://github.com/equinor/gordo",
    use_scm_version={"write_to": "gordo/_version.py", "relative_to": __file__},
    zip_safe=True,
    package_data={
        "": [
            "gordo/workflow/workflow_generator/resources/argo-workflow.yml.template",
            "gordo/machine/dataset/data_provider/resources/assets_config.yaml"
        ]
    },
    include_package_data=True,
)
