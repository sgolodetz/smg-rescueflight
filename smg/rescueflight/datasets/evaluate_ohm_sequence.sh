#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -ne 1 ]
then
  echo "Usage: evaluate_ohm_sequence.sh <sequence name>"
  exit 1
fi

# Reconstruct the scene and the people in it in various different ways.
./reconstruct_ohm_sequence.sh "$1"

# Enable the conda command.
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE\\etc\\profile.d\\conda.sh"

# Determine the sequence directory.
sequence_dir=`./determine_sequence_dir.sh ohm "$1" true`

# Evaluate the 3D skeletons for each of the different methods in turn.
for detector_tag in lcrnet xnect
do
  echo "Evaluating skeletons for $1 ($detector_tag)"
  output_filename="skeleton_metrics-$detector_tag.txt"
  if [ -e "$sequence_dir/people/$output_filename" ]
  then
    echo "- Found $output_filename: skipping"
  elif [ -e "$sequence_dir/people/$detector_tag" ]
  then
    conda activate smglib
    python ../vicon/evaluate_skeleton_sequence_vs_vicon.py --batch -s "$sequence_dir" -t "$detector_tag"  > "$sequence_dir/people/$output_filename" 2>/dev/null
    echo "- Written metrics to $output_filename"
    conda deactivate
  else
    echo "- Missing skeleton detection results for $detector_tag"
  fi
done
