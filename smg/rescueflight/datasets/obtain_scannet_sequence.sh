#! /bin/bash -e

if [ $# -ne 1 ]
then
  echo "Usage: obtain_scannet_sequence.sh <sequence name>"
  exit 1
fi

sequence_dir=`./determine_scannet_sequence_dir.sh "$1" true`
if [ -e "$sequence_dir" ]
then
  echo "Sequence already available: $1"
  exit 1
fi

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE\\etc\\profile.d\\conda.sh"
conda activate python2.7

root_dir=`./determine_scannet_root_dir.sh`
cd "$root_dir"

temp_dir="$root_dir/temp/scans_test/$1"
if [ -e "$temp_dir" ]
then
  echo "Already downloaded $1: skipping"
else
  echo "Downloading $1..."
  python download-scannet-noprompt.py -o temp --id="$1"
fi

if [ -e "$temp_dir/exported" ]
then
  echo "Already exported $1: skipping"
else
  echo "Exporting $1..."
  cd "$temp_dir"
  python /c/ScanNet/SensReader/python/reader.py --filename "$1.sens" --output_path exported --export_depth_images --export_color_images --export_poses --export_intrinsics
  cd -
fi

mv "$temp_dir/exported" "$sequence_dir"
/bin/rm -fR temp
