#! /bin/bash

# Check that the script is being used correctly.
if [ $# -ne 0 ]
then
  echo "Usage: determine_scannet_root_dir.sh"
  exit 1
fi

# Check various different possibilities for the ScanNet root directory, and echo the right one if found.
root_dir="/c/datasets/scannet"
if [ -e "$root_dir" ]
then
  echo "$root_dir"
  exit 0
fi

root_dir="/d/datasets/scannet"
if [ -e "$root_dir" ]
then
  echo "$root_dir"
  exit 0
fi
