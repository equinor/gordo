from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = [req.strip() for req in f.readlines()]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

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
    version='0.1.0',
    zip_safe=True,
)
