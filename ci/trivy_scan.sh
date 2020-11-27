#!/bin/bash

set -e

image=$1
if [ -z "$image" ]; then
    echo "Usage: $0 <image>"
    exit 1
fi

uname=$(uname -s 2>/dev/null)
trivy=$(which trivy || echo "")
if [ -z "$trivy" ]; then
    if [ "$uname"  == "Linux" ]; then
        sudo apt-get install wget apt-transport-https gnupg lsb-release
        wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
        echo deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main | sudo tee -a /etc/apt/sources.list.d/trivy.list
        sudo apt-get update
        sudo apt-get install trivy
    else
        echo "Unable to determine platform '$uname'" 
        exit 1
    fi
fi

trivy_version=$(trivy -v 2>/dev/null | head -1 | cut -d ' ' -f 2);

echo "Trivy version is ${trivy_version} and platform is ${uname}"
echo "Scanning image - '$image'"

trivy --clear-cache 
trivy --exit-code 10 -severity HIGH,CRITICAL --light --no-progress --ignore-unfixed --timeout "5m" "$image"
