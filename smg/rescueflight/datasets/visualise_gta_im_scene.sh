#! /bin/bash -e
if [ $# -lt 1 ]
then
  echo "Usage: reconstruct_gta_im_scene.sh <scene name> [args]"
  exit 1
fi

sequence_dir="/c/datasets/gta-im/$1"

if [ ! -e "$sequence_dir" ]
then
  echo "No such sequence: $sequence_dir"
  exit 1
fi

trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT

echo "Initialising skeleton detection services..."
python run_gta_im_skeleton_detection_service.py -s "$sequence_dir" > /dev/null 2>&1 &
python run_gta_im_skeleton_detection_service.py -s "$sequence_dir" -p 7853 > /dev/null 2>&1 &
sleep 5
echo "Initialising mapping server..."
# --max_depth=5.0 --octree_voxel_size=0.1 --tsdf_voxel_size=0.1
python ../mapping/scripts/run_octomap_mapping_server.py --detect_skeletons --use_received_depth --use_tsdf "${@:2}" > /dev/null 2>& 1 &
sleep 5
echo "Visualising $1..."
python run_gta_im_client.py -s "$sequence_dir"
