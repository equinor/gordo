#/usr/bin/env bash

set -e

PYTEST_ARG="-m 'not dockertest'"

function show_help() {
    echo "Usage: $0 [-h] ACTION"
    echo
    echo "Runs CI pytest action."
    echo
    echo "-h     display this help and exit"
    exit $1
}


while getopts "h" opt; do
  case "$opt" in
    h)
        show_help 0
        ;;
  esac
done

shift $((OPTIND-1))

action=$1

if [[ ! "$action" =~ ^[a-z-]*$ ]]; then
    echo "Wrong action '$action' format." 1>&2
    show_help 1
fi

export PYTHONPATH=.

case "$action" in
    all)
        pytest -n auto "$PYTEST_ARG" --ignore benchmarks
        ;;
    builder)
        pytest "$PYTEST_ARG" tests/gordo/builder
        ;;
    cli)
        pytest "$PYTEST_ARG" tests/gordo/cli
        ;;
    client)
        pytest "$PYTEST_ARG" tests/gordo/client
        ;;
    machine)
        pytest "$PYTEST_ARG" tests/gordo/machine
        ;;
    reporters)
        pytest "$PYTEST_ARG" tests/gordo/reporters
        ;;
    serializer)
        pytest "$PYTEST_ARG" tests/gordo/serializer
        ;;
    server)
        pytest "$PYTEST_ARG" tests/gordo/server
        ;;
    util)
        pytest "$PYTEST_ARG" tests/gordo/util
        ;;
    workflow)
        pytest "$PYTEST_ARG" tests/gordo/workflow
        ;;
    formatting)
        pytest "$PYTEST_ARG" tests/test_formatting.py
        ;;
    docker)
        pytest "$PYTEST_ARG" -m dockertest
        ;;
    allelse)
        pytest "$PYTEST_ARG" --ignore tests/gordo/builder \
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
    benchmarks) 
        pytest --benchmark-only benchmarks/
        ;;
    *)
        echo "Wrong action '$action'." 1>&2
        show_help 2
        ;;
esac
