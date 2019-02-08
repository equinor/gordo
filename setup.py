from setuptools import setup, find_packages

# Install requirements
install_requires = [
    "Click~=7.0",
    "h5py~=2.8",
    "influxdb~=5.2",
    "joblib~=0.13",
    "Keras~=2.2",
    "numpy~=1.15",
    "pandas~=0.23",
    "numexpr>=2.6.1",
    "pip-tools~=3.1",
    "python-dateutil~=2.7",
    "pyyaml>=4.2b1",
    "requests~=2.20",
    "scikit-learn~=0.20",
    "tensorflow~=1.12",
    "Flask~=1.0",
    "flask-restplus~=0.12",
    "azure-datalake-store~=0.0",
]

setup_requirements = ["pytest-runner", "setuptools_scm", "pytest-xdist==1.26.1"]

# Test requirements
test_requirements = [
    "docker==3.6.0",
    "pytest==4.0.0",
    "ruamel.yaml==0.15.76",
    "pytest-mypy==0.3.2",
    "responses==0.10.5",
    "black==18.9b0",
    "pytest-flakes~=4.0",
    "adal~=1.2",
    "jupyter==1.0.0",
    "notebook==5.7.4",
    "nbconvert==5.4.0",
    "tornado>=4, <6",  # https://github.com/jupyter/notebook/pull/4310#issuecomment-452368841
]

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
    install_requires=install_requires,
    license="Unlicense",
    name="gordo-components",
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/Statoil/gordo-flow",
    use_scm_version={
        "write_to": "gordo_components/_version.py",
        "relative_to": __file__,
    },
    zip_safe=True,
)
