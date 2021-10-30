#! /bin/bash -e

if [ $# -lt 3 ]
then
  echo "Usage: reconstruct_scannet_scene.sh <sequence name> <tag> {gt|ours} [args]"
  exit 1
fi

sequence_dir=`./determine_scannet_sequence_dir.sh "$1"`
if [ -z "$sequence_dir" ]
then
  echo "No such sequence: $1"
  exit 1
fi

if [ "$3" = "gt" ]
then
  echo "Reconstructing $1 ($2)"
  echo "- Running mapping server..."
  python ../mapping/scripts/run_open3d_mapping_server.py --batch --debug -p wait --output_dir="$sequence_dir/recon" --reconstruction_filename="$2.ply" --save_reconstruction --use_received_depth "${@:4}" > /dev/null 2>&1 &
  sleep 5
  echo "- Running mapping client..."
  python run_scannet_client.py --batch --canonicalise_poses -s "$sequence_dir" > /dev/null 2>&1
elif [ "$3" = "ours" ]
then
  echo "Reconstructing $1 ($2)"
  echo "- Running mapping server..."
  python ../mapping/scripts/run_open3d_mapping_server.py --batch --debug -p wait --output_dir="$sequence_dir/recon" --reconstruction_filename="$2.ply" --save_reconstruction "${@:4}" > /dev/null 2>&1 &
  sleep 5
  echo "- Running mapping client..."
  python run_scannet_client.py --batch -s "$sequence_dir" --use_tracker > /dev/null 2>&1
fi

echo "- Written reconstruction to: $sequence_dir/recon/$2.ply"
