#!/bin/bash

echo "Start"
"$@" &
child_pid=$!
while ps | grep $child_pid ; do
  ps -u
  sleep 30
done
echo "End"
