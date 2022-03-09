#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -lt 4 ]
then
  echo "Usage: reconstruct_ohm_scene_offline.sh <sequence name> <method tag> {dvmvs|mvdepth} {lcrnet|lcrnet-smpl|maskrcnn|nomask|xnect|xnect-smpl} [args]"
  exit 1
fi

# Check that the sequence directory exists.
sequence_dir=`./determine_sequence_dir.sh ohm "$1"`
if [ -z "$sequence_dir" ]
then
  echo "No such sequence: $1"
  exit 1
fi

# Start the reconstruction.
echo "Reconstructing $1 ($2)"

# If the output file already exists, early out.
if [ -f "$sequence_dir/reconstruction/$2.ply" ]
then
  echo "- Found $2.ply: skipping"
  exit 0
fi

# Enable the conda command.
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE\\etc\\profile.d\\conda.sh"

# If needed, run the people masking service.
echo "- Running people masking service..."
if [ "$4" == "nomask" ]
then
  detect_skeletons_flag=""
else
  ../skeletons/scripts/run_skeleton_detection_service.sh "$4" > /dev/null 2>&1 &
  pms_pid="$!"
  detect_skeletons_flag="--detect_skeletons"
fi

# Wait for the people masking service to initialise (if we're running it).
sleep 5

# Run the mapping server.
echo "- Running mapping server..."
conda activate smglib
echo "$detect_skeletons_flag" | xargs python ../mapping/scripts/run_open3d_mapping_server.py --batch --debug --output_dir="$sequence_dir/reconstruction" -p wait --reconstruction_filename="$2.ply" --save_reconstruction --depth_estimator_type="$3" "${@:5}" > /dev/null 2>&1 &
server_pid="$!"
conda deactivate

# Wait for the mapping server to initialise.
sleep 5

# Run the mapping client.
echo "- Running mapping client..."
conda activate smglib
python ../vicon/run_vicon_disk_client.py --batch -s "$sequence_dir" --use_tracked_poses --use_vicon_scale > /dev/null 2>&1
conda deactivate

# Wait for the mapping server to finish.
echo "- Writing reconstruction to: $sequence_dir/reconstruction/$2.ply"
wait "$server_pid"

# If the people masking service is running, ruthlessly kill it, since it will otherwise run forever.
if [ "$4" != "nomask" ]
then
  for p in $(ps | perl -lane 'if($F[1] eq '"$pms_pid"') { print $F[0]; }'); do kill -9 "$p"; done || true
  kill -9 "$pms_pid"
fi
