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
else
  echo "Found $1"
fi

tag="gt_4m_gt"
if [ -f "$sequence_dir/recon/$tag.ply" ]
then
  echo "- Found $tag: skipping"
else
  ./reconstruct_scannet_scene.sh "$1" "$tag" gt gt --max_depth=4.0
fi

tag="dvmvs_4m_gt"
if [ -f "$sequence_dir/recon/$tag.ply" ]
then
  echo "- Found $tag: skipping"
else
  ./reconstruct_scannet_scene.sh "$1" "$tag" dvmvs gt --max_depth=4.0 --no_depth_postprocessing
fi

tag="dvmvs_pp_4m_gt"
if [ -f "$sequence_dir/recon/$tag.ply" ]
then
  echo "- Found $tag: skipping"
else
  ./reconstruct_scannet_scene.sh "$1" "$tag" dvmvs gt --max_depth=4.0
fi

tag="mvdepth_4m_gt"
if [ -f "$sequence_dir/recon/$tag.ply" ]
then
  echo "- Found $tag: skipping"
else
  ./reconstruct_scannet_scene.sh "$1" "$tag" mvdepth gt --max_depth=4.0 --no_depth_postprocessing
fi

tag="mvdepth_pp_4m_gt"
if [ -f "$sequence_dir/recon/$tag.ply" ]
then
  echo "- Found $tag: skipping"
else
  ./reconstruct_scannet_scene.sh "$1" "$tag" mvdepth gt --max_depth=4.0
fi
