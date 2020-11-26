#!/bin/bash

set -e

image=$1
if [ -z "$image" ]; then
    echo "Usage: $0 <image>"
    exit 1
fi

uname=$(uname -s 2>/dev/null)
trivy=$(which trivy);
if [ -z "$trivy" ]; then
    if [ "$uname" == "Darwin" ]; then
        machine="macOS"
    elif [ "$uname"  == "Linux" ]; then
        machine="Linux"
    else
        echo "Unable to determine platform '$uname'" 
        exit 1
    fi
    TRIVY_VERSION=$(curl --silent "https://api.github.com/repos/aquasecurity/trivy/releases/latest" | grep '"tag_name":' | sed -E 's/.*"v([^"]+)".*/\1/');
    echo "Downloading trivy..";
    if [ -n "$TRIVY_VERSION" ] && [ -n "$machine" ]; then
        curl -Ls "https://github.com/aquasecurity/trivy/releases/download/v${TRIVY_VERSION}/trivy_${TRIVY_VERSION}_${machine}-64bit.tar.gz" 
        if tar zx --wildcards '*trivy'; then 
            echo "Download or extract failed for '${machine}' version '${TRIVY_VERSION}'."
            exit 1
        fi
        trivy="./trivy";
    fi
else
    TRIVY_VERSION=$(trivy -v 2>/dev/null | head -1 | cut -d ' ' -f 2);
fi

echo "Trivy version is ${TRIVY_VERSION} and platform is ${uname}"
echo "Scanning image - '$image'"

$trivy --clear-cache 
$trivy --exit-code 10 -severity HIGH,CRITICAL --light --no-progress --ignore-unfixed "$image"
