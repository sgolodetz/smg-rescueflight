#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -ne 1 ]
then
  echo "Usage: evaluate_gta_im_sequence.sh <sequence name>"
  exit 1
fi

# Reconstruct the scene and the people in it using the various different methods we want to compare.
./reconstruct_gta_im_sequence.sh "$1"

# TODO: Evaluate the scene reconstructions for each of the different methods in turn.
for method_tag in lcrnet maskrcnn xnect
do
    ./evaluate_scene.sh gta-im "$1" gt_"$method_tag" gt_gt
    ./evaluate_scene.sh gta-im "$1" gt_gt gt_"$method_tag"
    for percent_to_stop in 20 40 60 80
    do
        ./evaluate_scene.sh gta-im "$1" gt_"$method_tag"_"$percent_to_stop" gt_gt_"$percent_to_stop"
        ./evaluate_scene.sh gta-im "$1" gt_gt_"$percent_to_stop" gt_"$method_tag"_"$percent_to_stop"
    done
done

# Enable the conda command.
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE\\etc\\profile.d\\conda.sh"

# Determine the sequence directory.
sequence_dir=`./determine_sequence_dir.sh gta-im "$1" true`

# Evaluate the people masks for each of the different methods in turn.
for generator_tag in lcrnet maskrcnn xnect
do
  echo "Evaluating people masks for $1 ($generator_tag)"
  output_filename="people_mask_metrics-$generator_tag.txt"
  if [ -e "$sequence_dir/people/$output_filename" ]
  then
    echo "- Found $output_filename: skipping"
  elif [ -e "$sequence_dir/people/$generator_tag" ]
  then
    conda activate smglib
    python evaluate_people_mask_sequence.py -s "$sequence_dir" -t "$generator_tag" > "$sequence_dir/people/$output_filename"
    echo "- Written metrics to $output_filename"
    conda deactivate
  else
    echo "- Missing people mask results for $generator_tag"
  fi
done

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
    python evaluate_skeleton_sequence.py --batch -s "$sequence_dir" -t "$detector_tag" > "$sequence_dir/people/$output_filename" 2>/dev/null
    echo "- Written metrics to $output_filename"
    conda deactivate
  else
    echo "- Missing skeleton detection results for $detector_tag"
  fi
done
