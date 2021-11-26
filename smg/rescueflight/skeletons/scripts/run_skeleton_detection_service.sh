#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -lt 1 ]
then
  echo "Usage: run_skeleton_detection_service.sh {lcrnet|lcrnet-smpl|maskrcnn|xnect} [args]"
  exit 1
fi

# Enable the conda command.
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE\\etc\\profile.d\\conda.sh"

# Run the skeleton detection service.
if [ "$1" == "lcrnet" ] || [ "$1" == "lcrnet-smpl" ]
then
  conda activate lcrnet
  export PYTHONPATH="C:/smglib/smg-lcrnet/smg/external/lcrnet/Detectron.pytorch/lib;$PYTHONPATH"
  export PYTHONPATH="C:/smglib/smg-lcrnet/smg/external/lcrnet;$PYTHONPATH"
  export PYTHONPATH="C:/smglib/smg-lcrnet;$PYTHONPATH"

  if [ "$1" == "lcrnet-smpl" ]
  then
    python /c/smglib/smg-lcrnet/scripts/run_lcrnet_skeleton_detection_service.py -p 7854 &
  else
    python /c/smglib/smg-lcrnet/scripts/run_lcrnet_skeleton_detection_service.py "${@:2}"
  fi

  export PYTHONPATH=
  conda deactivate

  if [ "$1" == "lcrnet-smpl" ]
  then
    sleep 5
    conda activate smglib
    python ../smplx/run_smpl_skeleton_detection_service.py "${@:2}"
    conda deactivate
  fi
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
