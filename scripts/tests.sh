#/usr/bin/env bash

set -e

function show_help() {
    echo "Usage: $0 [-h] ACTION"
    echo
    echo "Runs CI pytest action."
    echo
    echo "-n     uses xdist to speedup slow-running tests"
    echo "-p     export PYTHONPATH=. environment variable. Helpful when gordo is not installed in the system"
    echo "-h     display this help and exit"
    exit $1
}


while getopts "nph" opt; do
    case "$opt" in
        n)
            use_xdist="true"
            ;;
        p)
            export_pythonpath="true"
            ;;
        h)
            show_help 0
            ;;
    esac
done

shift $((OPTIND-1))

action=$1

if [ -n "$export_pythonpath" ]; then
    export PYTHONPATH=.
fi

if [ -n "$use_xdist" ]; then
    slow_args="-n auto"
else
    slow_args="-n 0"
fi

case "$action" in
    all)
        pytest $slow_args -m "not dockertest" --ignore benchmarks
        ;;
    builder)
        pytest $slow_args -m "not dockertest" tests/gordo/builder
        ;;
    cli)
        pytest $slow_args -m "not dockertest" tests/gordo/cli
        ;;
    machine)
        pytest $slow_args -m "not dockertest" tests/gordo/machine
        ;;
    server)
        pytest $slow_args -m "not dockertest" tests/gordo/server
        ;;
    reporters)
        pytest -m "not dockertest" tests/gordo/reporters
        ;;
    serializer)
        pytest -m "not dockertest" tests/gordo/serializer
        ;;
    client)
        pytest -m "not dockertest" tests/gordo/client
        ;;
    util)
        pytest -m "not dockertest" tests/gordo/util
        ;;
    workflow)
        pytest -m "not dockertest" tests/gordo/workflow
        ;;
    formatting)
        pytest -m "not dockertest" tests/test_formatting.py
        ;;
    allelse)
        pytest -m "not dockertest" --ignore tests/gordo/builder \
            --ignore tests/gordo/cli \
            --ignore tests/gordo/client \
            --ignore tests/gordo/machine \
            --ignore tests/gordo/reporters \
            --ignore tests/gordo/serializer \
            --ignore tests/gordo/server \
            --ignore tests/gordo/util \
            --ignore tests/gordo/watchman \
            --ignore tests/gordo/workflow \
            --ignore tests/test_formatting.py \
            --ignore benchmarks \
            .
        ;;
    docker)
        pytest -m "dockertest" -m dockertest
        ;;
    benchmarks) 
        pytest --benchmark-only benchmarks/
        ;;
    *)
        echo "Wrong action \"$action\"." 1>&2
        show_help 2
        ;;
esac
