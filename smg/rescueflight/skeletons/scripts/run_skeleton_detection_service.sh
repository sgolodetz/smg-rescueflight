#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -lt 1 ]
then
  echo "Usage: run_skeleton_detection_service.sh {lcrnet|maskrcnn|xnect} [args]"
  exit 1
fi

# Enable the conda command.
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE\\etc\\profile.d\\conda.sh"

# Run the skeleton detection service.
if [ "$1" == "lcrnet" ]
then
  conda activate lcrnet
  export PYTHONPATH="C:/smglib/smg-lcrnet/smg/external/lcrnet/Detectron.pytorch/lib;$PYTHONPATH"
  export PYTHONPATH="C:/smglib/smg-lcrnet/smg/external/lcrnet;$PYTHONPATH"
  export PYTHONPATH="C:/smglib/smg-lcrnet;$PYTHONPATH"
  python /c/smglib/smg-lcrnet/scripts/run_lcrnet_skeleton_detection_service.py "${@:2}"
  conda deactivate
elif [ "$1" == "maskrcnn" ]
then
  conda activate smglib
  python ../detectron2/run_mask_rcnn_skeleton_detection_service.py "${@:2}"
  conda deactivate
elif [ "$1" == "xnect" ]
then
  conda activate xnect
  python /c/smglib/smg-pyxnect/scripts/run_xnect_skeleton_detection_service.py "${@:2}"
  conda deactivate
fi
