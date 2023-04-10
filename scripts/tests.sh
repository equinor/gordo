#/usr/bin/env bash

set -e

TESTS_DIR=tests/gordo
PYTEST_ARGS="-n auto"

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

case "$action" in
    all)
        pytest -n auto -m 'not dockertest' --ignore benchmarks
        ;;
    builder)
        pytest $PYTEST_ARGS $TESTS_DIR/builder
        ;;
    cli)
        pytest $PYTEST_ARGS $TESTS_DIR/cli
        ;;
    client)
        pytest $PYTEST_ARGS $TESTS_DIR/client
        ;;
    machine)
        pytest $PYTEST_ARGS $TESTS_DIR/machine
        ;;
    reporters)
        pytest $PYTEST_ARGS $TESTS_DIR/reporters
        ;;
    serializer)
        pytest $PYTEST_ARGS $TESTS_DIR/serializer
        ;;
    server)
        pytest $PYTEST_ARGS $TESTS_DIR/server
        ;;
    util)
        pytest $PYTEST_ARGS $TESTS_DIR/util
        ;;
    workflow)
        pytest $PYTEST_ARGS $TESTS_DIR/workflow
        ;;
    docker)
        pytest -m 'dockertest'
        ;;
    allelse)
        pytest $PYTEST_ARGS --ignore $TESTS_DIR/builder \
            --ignore $TESTS_DIR/cli \
            --ignore $TESTS_DIR/client \
            --ignore $TESTS_DIR/machine \
            --ignore $TESTS_DIR/reporters \
            --ignore $TESTS_DIR/serializer \
            --ignore $TESTS_DIR/server \
            --ignore $TESTS_DIR/util \
            --ignore $TESTS_DIR/watchman \
            --ignore $TESTS_DIR/workflow \
            --ignore tests/test_formatting.py \
            --ignore benchmarks
        ;;
    benchmarks) 
        pytest --benchmark-only benchmarks/
        ;;
    *)
        echo "Wrong action '$action'." 1>&2
        show_help 2
        ;;
esac
