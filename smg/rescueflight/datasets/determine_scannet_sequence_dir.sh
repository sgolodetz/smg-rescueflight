#! /bin/bash

# Check that the script is being used correctly.
if [ $# -ne 1 ] && [ $# -ne 2 ]
then
  echo "Usage: determine_scannet_sequence_dir.sh <scene name> [allow missing]"
  exit 1
fi

# Try to determine the location of the ScanNet root directory.
root_dir=`./determine_scannet_root_dir.sh`

# If the root directory was found:
if [ -e "$root_dir" ]
then
  # Derive the location of the sequence directory from the root directory.
  sequence_dir="$root_dir/$1"

  # If the sequence directory either exists, or is allowed not to exist, echo it.
  if [ -e "$sequence_dir" ] || [ $# -eq 2 ]
  then
    echo "$sequence_dir"
  fi
fi
