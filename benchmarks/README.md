### Benchmarks

Write benchmark tests using `pytest-benchmark`'s `benchmark` fixture.

[pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/en/latest/)

Run only these tests via `python setup.py testbenchmarks`, otherwise
these tests are not ran as part of continuous integration, and therefore
should not be used for testing functionality but speed.
