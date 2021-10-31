#! /bin/bash -e

if [ $# -ne 1 ]
then
  echo "Usage: evaluate_scannet_scenes.sh <sequence name>"
  exit 1
fi

sequence_dir=`./determine_scannet_sequence_dir.sh "$1"`
if [ -z "$sequence_dir" ]
then
  echo "No such sequence: $1"
  exit 1
fi

for tag in dvmvs_4m_gt dvmvs_pp_4m_gt mvdepth_4m_gt mvdepth_pp_4m_gt
do
  ./evaluate_scannet_scene.sh "$1" "$tag" gt_4m_gt
done
