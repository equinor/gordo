from os import path
from setuptools import setup, find_packages

# Install requirements
with open('requirements.txt', 'r') as f:
    requirements = [req.strip() for req in f.readlines()]

setup_requirements = ['pytest-runner', 'setuptools_scm']

# Test requirements
test_requirements = ['pytest', 'ruamel.yaml==0.15.76', 'pytest-mypy==0.3.2']

# Need the model server runtime requirements to run model tests
runtime_req_txt = path.join(
    path.dirname(__file__), 'gordo_components', 'runtime', 'requirements.txt'
)
with open(runtime_req_txt) as f:
    test_requirements.extend([req.strip() for req in f.readlines()])

setup(
    author="Miles Granger",
    author_email='milg@equinor.no',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Unlicense',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Train and build models for Argo / Kubernetes",
    entry_points={
        'console_scripts': [
            'gordo-components=gordo_components.cli:gordo',
        ],
    },
    install_requires=requirements,
    license="Unlicense",
    name='gordo-components',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Statoil/gordo-flow',
    use_scm_version={
        'write_to': 'gordo_components/_version.py',
        'relative_to': __file__
    },
    zip_safe=True,
)
