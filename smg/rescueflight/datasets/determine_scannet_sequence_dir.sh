#! /bin/bash

if [ $# -ne 1 ]
then
  echo "Usage: determine_scannet_sequence_dir.sh <scene name>"
  exit 1
fi

sequence_dir="/c/datasets/scannet/$1"
if [ -e "$sequence_dir" ]
then
  echo "$sequence_dir"
  exit 0
fi

sequence_dir="/d/datasets/scannet/$1"
if [ -e "$sequence_dir" ]
then
  echo "$sequence_dir"
  exit 0
fi
