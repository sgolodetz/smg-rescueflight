#! /bin/bash

# Check that the script is being used correctly.
if [ $# -ne 1 ]
then
  echo "Usage: determine_dataset_root_dir.sh <dataset name>"
  exit 1
fi

# Check various different possibilities for the dataset root directory, and echo the right one if found.
root_dir="/c/datasets/$1"
if [ -e "$root_dir" ]
then
  echo "$root_dir"
  exit 0
fi

root_dir="/d/datasets/$1"
if [ -e "$root_dir" ]
then
  echo "$root_dir"
  exit 0
fi
