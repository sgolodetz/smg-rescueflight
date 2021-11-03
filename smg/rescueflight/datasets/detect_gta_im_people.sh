#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -lt 4 ]
then
  echo "Usage: detect_gta_im_people.sh <sequence name> <method tag> {gt|lcrnet|maskrcnn} {gt|lcrnet|xnect} [args]"
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

# Start the people detection process.
echo "Detecting people for $1 ($2)"

# If the output directory already exists, early out.
if [ -e "$sequence_dir/people/$2" ]
then
  echo "- Found people/$2 directory: skipping"
  exit 0
fi

# Run the people masking service.
echo "- Running people masking service..."
# TODO: LCR-Net and Mask R-CNN.
if [ "$3" == "gt" ]
then
  conda activate smglib
  python run_gta_im_skeleton_detection_service.py -s "$sequence_dir" > /dev/null 2>&1 &
  pms_pid="$!"
  conda deactivate
fi

# Run the live skeleton detection service.
# TODO: LCR-Net and XNect.
echo "- Running live skeleton detection service..."
if [ "$4" == "gt" ]
then
  conda activate smglib
  python run_gta_im_skeleton_detection_service.py -s "$sequence_dir" -p 7853 > /dev/null 2>&1 &
  sds_pid="$!"
  conda deactivate
fi

# Wait for the services to initialise.
sleep 5

# Run the mapping server.
echo "- Running mapping server..."
conda activate smglib
python ../mapping/scripts/run_octomap_mapping_server.py --detect_skeletons -p wait --use_received_depth --octree_voxel_size=0.1 --output_dir="$sequence_dir/people/$2" "${@:5}" > /dev/null 2>& 1 &
conda deactivate

# Wait for the mapping server to initialise.
sleep 5

# Run the mapping client.
echo "- Running mapping client..."
conda activate smglib
python run_gta_im_client.py --batch -s "$sequence_dir"
conda deactivate

# Ruthlessly kill the people masking and live skeleton detection services, which would otherwise run foreover.
kill -9 "$pms_pid" "$sds_pid"
