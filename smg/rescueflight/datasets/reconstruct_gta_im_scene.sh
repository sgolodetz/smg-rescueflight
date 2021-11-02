#! /bin/bash -e

if [ $# -lt 2 ]
then
  echo "Usage: reconstruct_gta_im_scene.sh <scene name> {gt|ours} [args]"
  exit 1
fi

sequence_dir="/c/datasets/gta-im/$1"
if [ ! -e "$sequence_dir" ]
then
  sequence_dir="/d/datasets/gta-im/$1"
  if [ ! -e "$sequence_dir" ]
  then
    echo "No such sequence: $sequence_dir"
    exit 1
  fi
fi

# trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT

echo "Initialising skeleton detection service..."
python run_gta_im_skeleton_detection_service.py -s "$sequence_dir" > /dev/null 2>&1 &
sleep 5

if [ "$2" = "gt" ]
then
  echo "Initialising mapping server..."
  python ../mapping/scripts/run_open3d_mapping_server.py --batch --debug -p wait --detect_skeletons --output_dir="$sequence_dir/recon" --save_reconstruction --use_received_depth "${@:3}" > /dev/null 2>&1 &
  sleep 5
  echo "Reconstructing $1..."
  python run_gta_im_client.py --batch -s "$sequence_dir" > /dev/null 2>&1
  echo "Written reconstruction to: $sequence_dir/recon/mesh.ply"
elif [ "$2" = "ours" ]
then
  echo "Initialising mapping server..."
  python ../mapping/scripts/run_open3d_mapping_server.py --batch --debug -p wait --detect_skeletons --output_dir="$sequence_dir/recon" --save_reconstruction "${@:3}" > /dev/null 2>&1 &
  sleep 5
  echo "Reconstructing $1..."
  python run_gta_im_client.py --batch -s "$sequence_dir" > /dev/null 2>&1
  echo "Written reconstruction to: $sequence_dir/recon/mesh.ply"
fi
