#! /bin/bash -e

if [ $# -lt 1 ]
then
  echo "Usage: reconstruct_scannet_scenes.sh <sequence name>"
  exit 1
fi

sequence_dir=`./determine_scannet_sequence_dir.sh "$1"`
if [ -z "$sequence_dir" ]
then
  echo "No such sequence: $1"
  exit 1
fi

./reconstruct_scannet_scene.sh "$1" "gt_gt" gt gt --max_depth=20.0
./reconstruct_scannet_scene.sh "$1" "dvmvs_4m_gt" dvmvs gt --max_depth=4.0 --no_depth_postprocessing
./reconstruct_scannet_scene.sh "$1" "dvmvs_pp_4m_gt" dvmvs gt --max_depth=4.0
./reconstruct_scannet_scene.sh "$1" "mvdepth_4m_gt" mvdepth gt --max_depth=4.0 --no_depth_postprocessing
./reconstruct_scannet_scene.sh "$1" "mvdepth_pp_4m_gt" mvdepth gt --max_depth=4.0
