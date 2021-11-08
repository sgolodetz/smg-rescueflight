#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -lt 1 ]
then
  echo "Usage: reconstruct_gta_im_sequence.sh <sequence name>"
  exit 1
fi

# Reconstruct the sequence using the various different methods we want to compare.
./reconstruct_gta_im_scene_offline.sh "$1" "gt_gt" gt gt --max_depth=20.0 --voxel_size=0.05
# TODO: Other depth estimators.

# TODO: Comment here.
./export_gta_im_people.sh "$1" gt gt gt --max_depth=10.0 --save_people_masks --save_skeletons
