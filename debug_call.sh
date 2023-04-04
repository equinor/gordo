#!/bin/bash

echo "Start"
"$@" &
child_pid=$!
while ps | grep $child_pid ; do
  ps aux | grep $child_pid
  pstree -sp $child_pid
  ps aux | grep python
  sleep 30
done
echo "End"
