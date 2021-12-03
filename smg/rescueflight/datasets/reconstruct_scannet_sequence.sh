#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -lt 1 ]
then
  echo "Usage: reconstruct_scannet_sequence.sh <sequence name>"
  exit 1
fi

# Make sure that the sequence has been downloaded.
./obtain_scannet_sequence.sh "$1"

# Reconstruct the sequence using the various different methods we want to compare.
./reconstruct_scannet_scene.sh "$1" "gt_gt" gt gt --max_depth=20.0
./reconstruct_scannet_scene.sh "$1" "dvmvs_4m_gt" dvmvs gt --max_depth=4.0 --no_depth_postprocessing
./reconstruct_scannet_scene.sh "$1" "dvmvs_pp_4m_gt" dvmvs gt --max_depth=4.0
./reconstruct_scannet_scene.sh "$1" "mvdepth_4m_gt" mvdepth gt --max_depth=4.0 --no_depth_postprocessing
./reconstruct_scannet_scene.sh "$1" "mvdepth_pp_4m_gt" mvdepth gt --max_depth=4.0
./reconstruct_scannet_scene.sh "$1" "foo" dvmvs gt --max_depth=4.0
