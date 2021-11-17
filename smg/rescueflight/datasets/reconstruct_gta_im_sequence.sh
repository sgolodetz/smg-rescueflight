#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -lt 1 ]
then
  echo "Usage: reconstruct_gta_im_sequence.sh <sequence name>"
  exit 1
fi

# Make sure that the sequence has been downloaded.
./obtain_gta_im_sequence.sh "$1"

# Reconstruct the scene using the various different methods we want to compare.
./reconstruct_gta_im_scene_offline.sh "$1" "gt_skeleton_eval" gt gt --max_depth=20.0 --voxel_size=0.05
# TODO: Other depth estimators.

#./reconstruct_gta_im_scene_offline.sh "$1" "gt_gt" gt gt --max_depth=20.0 --voxel_size=0.025
#GTA_IM_CLIENT_FLAGS="--percent_to_stop=25" ./reconstruct_gta_im_scene_offline.sh "$1" "gt_gt_25" gt gt --max_depth=20.0 --voxel_size=0.025
#GTA_IM_CLIENT_FLAGS="--percent_to_stop=50" ./reconstruct_gta_im_scene_offline.sh "$1" "gt_gt_50" gt gt --max_depth=20.0 --voxel_size=0.025
#GTA_IM_CLIENT_FLAGS="--percent_to_stop=75" ./reconstruct_gta_im_scene_offline.sh "$1" "gt_gt_75" gt gt --max_depth=20.0 --voxel_size=0.025

#./reconstruct_gta_im_scene_offline.sh "$1" "gt_maskrcnn" gt maskrcnn --max_depth=20.0 --voxel_size=0.025
#GTA_IM_CLIENT_FLAGS="--percent_to_stop=25" ./reconstruct_gta_im_scene_offline.sh "$1" "gt_maskrcnn_25" gt maskrcnn --max_depth=20.0 --voxel_size=0.025
#GTA_IM_CLIENT_FLAGS="--percent_to_stop=50" ./reconstruct_gta_im_scene_offline.sh "$1" "gt_maskrcnn_50" gt maskrcnn --max_depth=20.0 --voxel_size=0.025
#GTA_IM_CLIENT_FLAGS="--percent_to_stop=75" ./reconstruct_gta_im_scene_offline.sh "$1" "gt_maskrcnn_75" gt maskrcnn --max_depth=20.0 --voxel_size=0.025

# Reconstruct the people in the scene using the various different methods we want to compare.
./reconstruct_gta_im_people.sh "$1" gt gt gt --max_depth=10.0 --save_people_masks --save_skeletons

if [ `./conda_env_exists.sh lcrnet` == "1" ]
then
  ./reconstruct_gta_im_people.sh "$1" lcrnet lcrnet gt --max_depth=10.0 --save_people_masks --save_skeletons
else
  echo "Cannot reconstruct people for $1 (lcrnet)"
fi

./reconstruct_gta_im_people.sh "$1" maskrcnn maskrcnn gt --max_depth=10.0 --save_people_masks

if [ `./conda_env_exists.sh xnect` == "1" ]
then
  ./reconstruct_gta_im_people.sh "$1" xnect xnect gt --max_depth=10.0 --save_people_masks --save_skeletons
else
  echo "Cannot reconstruct people for $1 (xnect)"
fi
