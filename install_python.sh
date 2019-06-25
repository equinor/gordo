#!/usr/bin/env bash

set -e

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
	brew update;
	brew install python$PYTHON_VERSION;
else
    apt update -y;
	apt install -y python$PYTHON_VERSION;
fi
