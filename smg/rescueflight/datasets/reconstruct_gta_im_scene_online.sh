#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -lt 4 ]
then
  echo "Usage: reconstruct_gta_im_scene_online.sh <sequence name> {batch|nobatch} {gt|lcrnet|lcrnet-smpl|maskrcnn|xnect|xnect-smpl} {gt|lcrnet|xnect} [args]"
  exit 1
fi

# Check that the sequence directory exists.
sequence_dir=`./determine_sequence_dir.sh gta-im "$1"`
if [ -z "$sequence_dir" ]
then
  echo "No such sequence: $1"
  exit 1
fi

# Enable the conda command.
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE\\etc\\profile.d\\conda.sh"

# Run the people masking service.
echo "- Running people masking service..."
if [ "$3" == "gt" ]
then
  conda activate smglib
  python run_gta_im_skeleton_detection_service.py -s "$sequence_dir" > /dev/null 2>&1 &
  pms_pid="$!"
  conda deactivate
else
  ../skeletons/scripts/run_skeleton_detection_service.sh "$3" > /dev/null 2>&1 &
  pms_pid="$!"
fi

# Run the live skeleton detection service.
echo "- Running live skeleton detection service..."
if [ "$4" == "gt" ]
then
  conda activate smglib
  python run_gta_im_skeleton_detection_service.py -s "$sequence_dir" -p 7853 > /dev/null 2>&1 &
  sds_pid="$!"
  conda deactivate
else
  ../skeletons/scripts/run_skeleton_detection_service.sh "$4" -p 7853 > /dev/null 2>&1 &
  sds_pid="$!"
fi

# Wait for the services to initialise.
sleep 5

# Run the mapping server.
echo "- Running mapping server..."
conda activate smglib
if [ "$2" == "batch" ]
then
  python ../mapping/scripts/run_octomap_mapping_server.py --batch --detect_skeletons --use_received_depth "${@:5}" > /dev/null 2>& 1 &
else
  python ../mapping/scripts/run_octomap_mapping_server.py --detect_skeletons --use_received_depth "${@:5}" > /dev/null 2>& 1 &
fi
server_pid="$!"
conda deactivate

# Wait for the mapping server to initialise.
sleep 5

# Run the mapping client.
echo "- Running mapping client..."
conda activate smglib
if [ "$2" == "batch" ]
then
  python run_gta_im_client.py --batch -s "$sequence_dir"
else
  python run_gta_im_client.py -s "$sequence_dir"
fi
conda deactivate

# Wait for the mapping server to finish.
wait "$server_pid"

# Ruthlessly kill the people masking and live skeleton detection services, which would otherwise run forever.
for p in $(ps | perl -lane 'if($F[1] eq '"$pms_pid"' || $F[1] eq '"$sds_pid"') { print $F[0]; }'); do kill -9 "$p"; done || true
kill -9 "$pms_pid" "$sds_pid"
