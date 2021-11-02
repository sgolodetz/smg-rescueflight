#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -lt 4 ]
then
  echo "Usage: reconstruct_scannet_scene.sh <sequence name> <method tag> {gt|dvmvs|mvdepth} {gt|track} [args]"
  exit 1
fi

# Check that the sequence directory exists.
sequence_dir=`./determine_scannet_sequence_dir.sh "$1"`
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

# Run the mapping server.
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

# Wait for the mapping server to initialise.
sleep 5

# Run the mapping client.
echo "- Running mapping client..."
if [ "$4" = "gt" ]
then
  python run_scannet_client.py --batch -s "$sequence_dir" --canonicalise_poses > /dev/null 2>&1
else
  python run_scannet_client.py --batch -s "$sequence_dir" --use_tracker > /dev/null 2>&1
fi

# Wait for the reconstruction to be written to disk.
echo "- Writing reconstruction to: $sequence_dir/recon/$2.ply"
sleep 5
