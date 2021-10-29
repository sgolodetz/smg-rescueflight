#! /bin/bash -e

if [ $# -lt 1 ]
then
  echo "Usage: reconstruct_scannet_scenes.sh <scene list file>"
  exit 1
fi

cat $1 | tr -d '\r' | while read f
do
  sequence_dir=`./determine_scannet_sequence_dir.sh "$f"`
  if [ -z "$sequence_dir" ]
  then
    echo "No such sequence: $f"
    continue
  else
    echo "Found $f"
  fi

  tag="gt_4m"
  if [ -f "$sequence_dir/recon/$tag.ply" ]
  then
    echo "- Found $tag: skipping"
  else
    ./reconstruct_scannet_scene.sh "$f" "$tag" gt --max_depth=4.0
  fi

  tag="dvmvs_raw_4m"
  if [ -f "$sequence_dir/recon/$tag.ply" ]
  then
    echo "- Found $tag: skipping"
  else
    ./reconstruct_scannet_scene.sh "$f" "$tag" ours --depth_estimator_type=dvmvs --max_depth=4.0 --no_depth_postprocessing
  fi

  tag="dvmvs_pp_4m"
  if [ -f "$sequence_dir/recon/$tag.ply" ]
  then
    echo "- Found $tag: skipping"
  else
    ./reconstruct_scannet_scene.sh "$f" "$tag" ours --depth_estimator_type=dvmvs --max_depth=4.0
  fi

  tag="mvdepth_raw_4m"
  if [ -f "$sequence_dir/recon/$tag.ply" ]
  then
    echo "- Found $tag: skipping"
  else
    ./reconstruct_scannet_scene.sh "$f" "$tag" ours --depth_estimator_type=mvdepth --max_depth=4.0 --no_depth_postprocessing
  fi

  tag="mvdepth_pp_4m"
  if [ -f "$sequence_dir/recon/$tag.ply" ]
  then
    echo "- Found $tag: skipping"
  else
    ./reconstruct_scannet_scene.sh "$f" "$tag" ours --depth_estimator_type=mvdepth --max_depth=4.0
  fi
done
