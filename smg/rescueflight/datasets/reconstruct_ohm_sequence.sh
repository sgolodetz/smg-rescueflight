#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -lt 1 ]
then
  echo "Usage: reconstruct_ohm_sequence.sh <sequence name>"
  exit 1
fi

# Reconstruct a version of the scene to show when performing the skeleton evaluation.
# TODO

# Reconstruct the people in the scene using the various different methods we want to compare.
for method_tag in lcrnet xnect
do
  if [ `./conda_env_exists.sh "$method_tag"` == "1" ]
  then
    ./reconstruct_ohm_people.sh "$1" "$method_tag" "$method_tag" --save_skeletons
  else
    echo "Cannot reconstruct people for $1 ($method_tag)"
  fi
done
