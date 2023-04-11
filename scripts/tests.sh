#/usr/bin/env bash

set -e

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

export PYTHONPATH=.

case "$action" in
    all)
        pytest -n auto -m "not dockertest" --ignore benchmarks
        ;;
    builder)
        pytest -m "not dockertest" tests/gordo/builder
        ;;
    cli)
        pytest -m "not dockertest" tests/gordo/cli
        ;;
    client)
        pytest -m "not dockertest" tests/gordo/client
        ;;
    machine)
        pytest -m "not dockertest" tests/gordo/machine
        ;;
    reporters)
        pytest -m "not dockertest" tests/gordo/reporters
        ;;
    serializer)
        pytest -m "not dockertest" tests/gordo/serializer
        ;;
    server)
        pytest -m "not dockertest" tests/gordo/server
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
        echo "Wrong action '$action'." 1>&2
        show_help 2
        ;;
esac
