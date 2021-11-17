#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -lt 4 ]
then
  echo "Usage: reconstruct_gta_im_scene_offline.sh <sequence name> <method tag> {gt|dvmvs|mvdepth} {gt|lcrnet|maskrcnn|xnect} [args]"
  exit 1
fi

# Check that the sequence directory exists.
sequence_dir=`./determine_sequence_dir.sh gta-im "$1"`
if [ -z "$sequence_dir" ]
then
  echo "No such sequence: $1"
  exit 1
fi

# Start the reconstruction.
echo "Reconstructing $1 ($2)"

# If the output file already exists, early out.
if [ -f "$sequence_dir/recon/$2.ply" ]
then
  echo "- Found $2.ply: skipping"
  exit 0
fi

# Enable the conda command.
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE\\etc\\profile.d\\conda.sh"

# Run the people masking service.
echo "- Running people masking service..."
if [ "$4" == "gt" ]
then
  conda activate smglib
  python run_gta_im_skeleton_detection_service.py -s "$sequence_dir" > /dev/null 2>&1 &
  pms_pid="$!"
  conda deactivate
else
  ../skeletons/scripts/run_skeleton_detection_service.sh "$4" > /dev/null 2>&1 &
  pms_pid="$!"
fi

# Wait for the people masking service to initialise.
sleep 5

# Run the mapping server.
echo "- Running mapping server..."
conda activate smglib
if [ "$3" == "gt" ]
then
  python ../mapping/scripts/run_open3d_mapping_server.py --batch --debug --detect_skeletons --output_dir="$sequence_dir/recon" -p wait --reconstruction_filename="$2.ply" --save_reconstruction --use_received_depth "${@:5}" > /dev/null 2>&1 &
else
  python ../mapping/scripts/run_open3d_mapping_server.py --batch --debug --detect_skeletons --output_dir="$sequence_dir/recon" -p wait --reconstruction_filename="$2.ply" --save_reconstruction --depth_estimator_type="$3" "${@:5}" > /dev/null 2>&1 &
fi
server_pid="$!"
conda deactivate

# Wait for the mapping server to initialise.
sleep 5

# Run the mapping client.
echo "- Running mapping client..."
conda activate smglib
echo "$GTA_IM_CLIENT_FLAGS" | xargs python run_gta_im_client.py --batch -s "$sequence_dir" > /dev/null 2>&1
conda deactivate

# Wait for the mapping server to finish.
echo "- Writing reconstruction to: $sequence_dir/recon/$2.ply"
wait "$server_pid"

# Ruthlessly kill the people masking service, which would otherwise run forever.
for p in $(ps | perl -lane 'if($F[1] eq '"$pms_pid"') { print $F[0]; }'); do kill -9 "$p"; done || true
kill -9 "$pms_pid"
