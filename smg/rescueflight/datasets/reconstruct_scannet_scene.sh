#! /bin/bash -e
if [ $# -lt 2 ]
then
  echo "Usage: reconstruct_scannet_scene.sh <scene name> {gt|ours} [args]"
  exit 1
fi

sequence_dir="/c/datasets/scannet/$1"

if [ ! -e "$sequence_dir" ]
then
  echo "No such sequence: $sequence_dir"
  exit 1
fi

if [ "$2" = "gt" ]
then
  echo "Initialising mapping server..."
  python /c/smglib/smg-rescueflight/smg/rescueflight/mapping/scripts/run_open3d_mapping_server.py --batch --debug -p wait --output_dir="$sequence_dir/recon" --save_reconstruction --use_received_depth "${@:3}" > /dev/null 2>&1 &
  sleep 10
  echo "Reconstructing $1..."
  python /c/smglib/smg-rescueflight/smg/rescueflight/datasets/run_scannet_client.py --batch -s "$sequence_dir" > /dev/null 2>&1
  echo "Written reconstruction to: $sequence_dir/recon/mesh.ply"
elif [ "$2" = "ours" ]
then
  echo "Initialising mapping server..."
  python /c/smglib/smg-rescueflight/smg/rescueflight/mapping/scripts/run_open3d_mapping_server.py --batch --debug -p wait --output_dir="$sequence_dir/recon" --save_reconstruction "${@:3}" > /dev/null 2>&1 &
  sleep 10
  echo "Reconstructing $1..."
  python /c/smglib/smg-rescueflight/smg/rescueflight/datasets/run_scannet_client.py --batch -s "$sequence_dir" --use_tracker > /dev/null 2>&1
  echo "Written reconstruction to: $sequence_dir/recon/mesh.ply"
fi
