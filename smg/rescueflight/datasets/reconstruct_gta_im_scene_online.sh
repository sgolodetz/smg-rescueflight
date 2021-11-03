#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -lt 4 ]
then
  echo "Usage: reconstruct_gta_im_scene_online.sh <sequence name> {batch|nobatch} {gt|lcrnet|maskrcnn} {gt|lcrnet|xnect} [args]"
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
elif [ "$3" == "lcrnet" ]
then
  conda activate lcrnet
  export PYTHONPATH="C:/smglib/smg-lcrnet/smg/external/lcrnet/Detectron.pytorch/lib;$PYTHONPATH"
  export PYTHONPATH="C:/smglib/smg-lcrnet/smg/external/lcrnet;$PYTHONPATH"
  export PYTHONPATH="C:/smglib/smg-lcrnet;$PYTHONPATH"
  python /c/smglib/smg-lcrnet/scripts/run_lcrnet_skeleton_detection_service.py > /dev/null 2>&1 &
  pms_pid="$!"
  export PYTHONPATH=
  conda deactivate
elif [ "$3" == "maskrcnn" ]
then
  conda activate smglib
  python ../detectron2/run_mask_rcnn_skeleton_detection_service.py > /dev/null 2>&1 &
  pms_pid="$!"
  conda deactivate
fi

# Run the live skeleton detection service.
# TODO: XNect.
echo "- Running live skeleton detection service..."
if [ "$4" == "gt" ]
then
  conda activate smglib
  python run_gta_im_skeleton_detection_service.py -s "$sequence_dir" -p 7853 > /dev/null 2>&1 &
  sds_pid="$!"
  conda deactivate
elif [ "$4" == "lcrnet" ]
then
  conda activate lcrnet
  export PYTHONPATH="C:/smglib/smg-lcrnet/smg/external/lcrnet/Detectron.pytorch/lib;$PYTHONPATH"
  export PYTHONPATH="C:/smglib/smg-lcrnet/smg/external/lcrnet;$PYTHONPATH"
  export PYTHONPATH="C:/smglib/smg-lcrnet;$PYTHONPATH"
  python /c/smglib/smg-lcrnet/scripts/run_lcrnet_skeleton_detection_service.py -p 7853 > /dev/null 2>&1 &
  sds_pid="$!"
  export PYTHONPATH=
  conda deactivate
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

# Ruthlessly kill the people masking and live skeleton detection services, which would otherwise run foreover.
kill -9 "$pms_pid" "$sds_pid"
