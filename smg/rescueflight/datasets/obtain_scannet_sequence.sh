#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -ne 1 ]
then
  echo "Usage: obtain_scannet_sequence.sh <sequence name>"
  exit 1
fi

# Start obtaining the sequence.
echo "Obtaining $1"

# If the sequence directory already exists, there's no need to obtain the sequence, so early out.
sequence_dir=`./determine_sequence_dir.sh scannet "$1" true`
if [ -e "$sequence_dir" ]
then
  echo "- Already obtained $1: skipping"
  exit 0
fi

# Download the sequence if necessary.
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE\\etc\\profile.d\\conda.sh"
conda activate python2.7

root_dir=`./determine_root_dir.sh scannet`
cd "$root_dir"

temp_dir="$root_dir/temp/scans_test/$1"
if [ -e "$temp_dir" ]
then
  echo "- Already downloaded $1: skipping"
else
  echo "- Downloading $1..."
  python download-scannet-noprompt.py -o temp --id="$1" > /dev/null 2>&1
  if [ ! -e "$temp_dir" ]
  then
    echo "- Download failed!"
    exit 1
  fi
fi

# Export the images, poses and camera intrinsics from the downloaded .sens file if necessary.
if [ -e "$temp_dir/exported" ]
then
  echo "- Already exported $1: skipping"
else
  echo "- Exporting $1..."
  cd "$temp_dir"
  python /c/ScanNet/SensReader/python/reader.py --filename "$1.sens" --output_path exported --export_depth_images --export_color_images --export_poses --export_intrinsics
  cd -
fi

# Move the directory containing the exported sequence into the correct place.
echo "- Finalising..."
mv "$temp_dir/exported" "$sequence_dir"

# Clean up.
/bin/rm -fR temp
