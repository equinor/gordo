#!/bin/bash

echo "Start"
"$@" &
child_pid=$!
while ps | grep $child_pid ; do
  if ps aux | grep $child_pid | grep Sl; then
      echo "ps aux | grep $child_pid"
      ps aux | grep $child_pid
      echo "pstree -sp $child_pid"
      pstree -sp $child_pid
      echo "timeout 1 strace -p $child_pid"
      sudo timeout 1 strace -p $child_pid
  fi
  sleep 30
done
echo "End"
