#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -lt 4 ]
then
  echo "Usage: reconstruct_gta_im_people.sh <sequence name> <method tag> {gt|lcrnet|maskrcnn} {gt|lcrnet|xnect} [args]"
  exit 1
fi

# Check that the sequence directory exists.
sequence_dir=`./determine_sequence_dir.sh gta-im "$1"`
if [ -z "$sequence_dir" ]
then
  echo "No such sequence: $1"
  exit 1
fi

# Start the people reconstruction process.
echo "Reconstructing people for $1 ($2)"

# If the output directory already exists, early out.
if [ -e "$sequence_dir/people/$2" ]
then
  echo "- Found people/$2 directory: skipping"
  exit 0
fi

# Otherwise, run the reconstruction.
./reconstruct_gta_im_scene_online.sh "$1" batch "$3" "$4" --octree_voxel_size=0.2 --output_dir="$sequence_dir/people/$2" -p wait "${@:5}"
