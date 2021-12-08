#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -ne 1 ]
then
  echo "Usage: evaluate_scannet_sequence.sh <sequence name>"
  exit 1
fi

# Reconstruct the scene using the various different methods we want to compare.
./reconstruct_scannet_sequence.sh "$1"

# Evaluate the reconstructions for each of the different methods in turn.
for method_tag in dvmvs_4m_gt dvmvs_pp_4m_gt mvdepth_4m_gt mvdepth_pp_4m_gt  # mvdepth2_4m_gt mvdepth2_pp_4m_gt
do
  ./evaluate_scene.sh scannet "$1" "$method_tag" gt_gt
  ./evaluate_scene.sh scannet "$1" gt_gt "$method_tag"
done
