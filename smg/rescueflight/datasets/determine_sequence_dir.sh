#! /bin/bash

# Check that the script is being used correctly.
if [ $# -ne 2 ] && [ $# -ne 3 ]
then
  echo "Usage: determine_sequence_dir.sh <dataset name> <sequence name> [allow missing]"
  exit 1
fi

# Try to determine the location of the dataset root directory.
root_dir=`./determine_root_dir.sh "$1"`

# If the root directory was found:
if [ -e "$root_dir" ]
then
  # Derive the location of the sequence directory from the root directory.
  sequence_dir="$root_dir/$2"

  # If the sequence directory either exists, or is allowed not to exist, echo it.
  if [ -e "$sequence_dir" ] || [ $# -eq 3 ]
  then
    echo "$sequence_dir"
  fi
fi
