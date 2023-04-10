#/usr/bin/env bash

set -e

NON_DOCKER_ARGS="-m 'not dockertest'"
DOCKER_ARGS="-m 'dockertest'"

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
        pytest -n auto $NON_DOCKER_ARGS --ignore benchmarks
        ;;
    builder)
        pytest $NON_DOCKER_ARGS tests/gordo/builder
        ;;
    cli)
        pytest $NON_DOCKER_ARGS tests/gordo/cli
        ;;
    client)
        pytest $NON_DOCKER_ARGS tests/gordo/client
        ;;
    machine)
        pytest $NON_DOCKER_ARGS tests/gordo/machine
        ;;
    reporters)
        pytest $NON_DOCKER_ARGS tests/gordo/reporters
        ;;
    serializer)
        pytest $NON_DOCKER_ARGS tests/gordo/serializer
        ;;
    server)
        pytest $NON_DOCKER_ARGS tests/gordo/server
        ;;
    util)
        pytest $NON_DOCKER_ARGS tests/gordo/util
        ;;
    workflow)
        pytest $NON_DOCKER_ARGS tests/gordo/workflow
        ;;
    formatting)
        pytest $NON_DOCKER_ARGS tests/test_formatting.py
        ;;
    docker)
        pytest $DOCKER_ARGS
        ;;
    allelse)
        pytest $NON_DOCKER_ARGS --ignore tests/gordo/builder \
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
