#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -lt 1 ]
then
  echo "Usage: reconstruct_ohm_sequence.sh <sequence name>"
  exit 1
fi

# Enable the conda command.
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE\\etc\\profile.d\\conda.sh"

# Determine the sequence directory.
sequence_dir=`./determine_sequence_dir.sh ohm "$1" true`

# If not already done, calculate the transformation from world space to Vicon space for the sequence.
echo "Calculating vTw for $1"
if [ -e "$sequence_dir/reconstruction/vicon_from_world.txt" ]
then
  echo "- Found vicon_from_world.txt: skipping"
else
  conda activate smglib
  python ../vicon/calculate_vicon_from_world.py -s "$sequence_dir" --save > /dev/null 2>&1
  conda deactivate
  echo "- Written vTw to vicon_from_world.txt"
fi

# Reconstruct the scene.
./reconstruct_ohm_scene_offline.sh "$1" world_mesh dvmvs maskrcnn --max_depth=4.0 --voxel_size=0.025

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
