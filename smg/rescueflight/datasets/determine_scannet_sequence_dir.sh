#! /bin/bash

if [ $# -ne 1 ] && [ $# -ne 2 ]
then
  echo "Usage: determine_scannet_sequence_dir.sh <scene name> [allow missing]"
  exit 1
fi

root_dir=`./determine_scannet_root_dir.sh`
if [ -e "$root_dir" ]
then
  sequence_dir="$root_dir/$1"
  if [ $# -eq 2 ] || [ -e "$sequence_dir" ]
  then
    echo "$sequence_dir"
    exit 0
  fi
fi
