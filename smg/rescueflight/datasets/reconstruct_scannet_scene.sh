#! /bin/bash -e

if [ $# -lt 4 ]
then
  echo "Usage: reconstruct_scannet_scene.sh <sequence name> <tag> {gt|dvmvs|mvdepth} {gt|track} [args]"
  exit 1
fi

sequence_dir=`./determine_scannet_sequence_dir.sh "$1"`
if [ -z "$sequence_dir" ]
then
  echo "No such sequence: $1"
  exit 1
fi

echo "Reconstructing $1 ($2)"

if [ -f "$sequence_dir/recon/$2.ply" ]
then
  echo "- Found $2.ply: skipping"
  exit 0
fi

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE\\etc\\profile.d\\conda.sh"
conda activate smglib

echo "- Running mapping server..."
if [ "$3" = "gt" ]
then
  python ../mapping/scripts/run_open3d_mapping_server.py --batch --debug --output_dir="$sequence_dir/recon" -p wait --reconstruction_filename="$2.ply" --save_reconstruction --use_received_depth "${@:5}" > /dev/null 2>&1 &
else
  python ../mapping/scripts/run_open3d_mapping_server.py --batch --debug --output_dir="$sequence_dir/recon" -p wait --reconstruction_filename="$2.ply" --save_reconstruction --depth_estimator_type="$3" "${@:5}" > /dev/null 2>&1 &
fi

sleep 5

echo "- Running mapping client..."
if [ "$4" = "gt" ]
then
  python run_scannet_client.py --batch -s "$sequence_dir" --canonicalise_poses > /dev/null 2>&1
else
  python run_scannet_client.py --batch -s "$sequence_dir" --use_tracker > /dev/null 2>&1
fi

echo "- Writing reconstruction to: $sequence_dir/recon/$2.ply"
sleep 5
