#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -lt 3 ]
then
  echo "Usage: visualise_gta_im_scene.sh <sequence name> {gt|lcrnet|maskrcnn} {gt|lcrnet|xnect} [args]"
  exit 1
fi

# Run the visualisation.
echo "Visualising $1"
./reconstruct_gta_im_scene_online.sh "$1" nobatch "${@:2}" --use_tsdf
