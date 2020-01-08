#!/usr/bin/env bash

# exit when any command fails
set -e

until mountpoint -q /gordo; do
    echo "$(date) - wainting for /gordo to be mounted..."
    sleep 1
done       

ls -l /gordo

gordo build

ls -l /gordo
