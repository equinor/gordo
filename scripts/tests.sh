#/usr/bin/env bash

set -e

TESTS_DIR=tests/gordo

action=$1

function show_help() {
    echo "Usage: $0 [-h] ACTION"
    echo
    echo "Runs CI pytest action."
    echo
    echo "-h "

    exit $1
}

while getopts "h" opt; do
  case "$opt" in
    h)
        show_help 0
        ;;
  esac
done

if [[ ! "$action" =~ ^[a-z-]*$ ]]; then
    echo "Wrong action '$action' format." 1>&2
    show_help 1
fi

case "$action" in
    all)
        pytest --ignore benchmarks
        ;;
    builder)
        pytest $TESTS_DIR/builder
        ;;
    cli)
        pytest $TESTS_DIR/cli
        ;;
    client)
        pytest $TESTS_DIR/client
        ;;
    machine)
        pytest $TESTS_DIR/machine
        ;;
    reporters)
        pytest $TESTS_DIR/reporters
        ;;
    serializer)
        pytest $TESTS_DIR/serializer
        ;;
    server)
        pytest $TESTS_DIR/server
        ;;
    util)
        pytest $TESTS_DIR/util
        ;;
    workflow)
        pytest $TESTS_DIR/workflow
        ;;
    docker)
        pytest -m 'dockertest'
        ;;
    allelse)
        pytest --ignore $TESTS_DIR/builder \
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
