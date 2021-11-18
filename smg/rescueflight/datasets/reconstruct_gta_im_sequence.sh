#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -lt 1 ]
then
  echo "Usage: reconstruct_gta_im_sequence.sh <sequence name>"
  exit 1
fi

# Make sure that the sequence has been downloaded.
./obtain_gta_im_sequence.sh "$1"

# Reconstruct the scene numerous times using different people masking approaches and stopping at different points.
./reconstruct_gta_im_scene_offline.sh "$1" gt_gt gt gt --max_depth=20.0 --voxel_size=0.025

for method_tag in lcrnet maskrcnn xnect
do
  if [ "$method_tag" == "maskrcnn" ] || [ `./conda_env_exists.sh "$method_tag"` == "1" ]
  then
    ./reconstruct_gta_im_scene_offline.sh "$1" gt_"$method_tag" gt "$method_tag" --max_depth=20.0 --voxel_size=0.025
    for percent_to_stop in 20 40 60 80
    do
        GTA_IM_CLIENT_FLAGS="--percent_to_stop=$percent_to_stop" ./reconstruct_gta_im_scene_offline.sh "$1" gt_"$method_tag"_"$percent_to_stop" gt "$method_tag" --max_depth=20.0 --voxel_size=0.025
    done
  else
    echo "Cannot reconstruct scenes for $1 ($method_tag)"
  fi
done

# Reconstruct a version of the scene to show when performing the skeleton evaluation.
./reconstruct_gta_im_scene_offline.sh "$1" "gt_skeleton_eval" gt gt --max_depth=20.0 --voxel_size=0.05

# Reconstruct the people in the scene using the various different methods we want to compare.
for method_tag in gt lcrnet maskrcnn xnect
do
  if [ "$method_tag" == "gt" ] || [ "$method_tag" == "maskrcnn" ] || [ `./conda_env_exists.sh "$method_tag"` == "1" ]
  then
    ./reconstruct_gta_im_people.sh "$1" "$method_tag" "$method_tag" gt --max_depth=10.0 --save_people_masks --save_skeletons
  else
    echo "Cannot reconstruct people for $1 ($method_tag)"
  fi
done
