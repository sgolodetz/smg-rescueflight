#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -ne 1 ]
then
  echo "Usage: evaluate_gta_im_sequence.sh <sequence name>"
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

# Repeatedly reconstruct the sequence using the various different methods we want to compare.
# TODO: Improve this comment (include something about skeletons and people masks).
./reconstruct_gta_im_sequence.sh "$1"

# Evaluate the reconstructions for each of the different methods in turn.
#for method_tag in dvmvs_4m_gt dvmvs_pp_4m_gt mvdepth_4m_gt mvdepth_pp_4m_gt
#do
#  ./evaluate_gta_im_scene.sh "$1" "$method_tag" gt_gt
#done

# Evaluate the people masks for each of the different methods in turn.
for generator_type in lcrnet maskrcnn
do
  echo "Evaluating people masks for $1 ($generator_type)"
  output_filename="people_mask_metrics-$generator_type.txt"
  if [ -e "$sequence_dir/people/$output_filename" ]
  then
    echo "- Found $output_filename: skipping"
  elif [ -e "$sequence_dir/people/$generator_type" ]
  then
    conda activate smglib
    python evaluate_gta_im_people_mask_sequence.py -s "$sequence_dir" -t "$generator_type" > "$sequence_dir/people/$output_filename"
    echo "- Written metrics to $output_filename"
    conda deactivate
  else
    echo "- Missing people mask results for $generator_type"
  fi
done

# Evaluate the 3D skeletons for each of the different methods in turn.
for detector_type in lcrnet xnect
do
  echo "Evaluating skeletons for $1 ($detector_type)"
  output_filename="skeleton_metrics-$detector_type.txt"
  if [ -e "$sequence_dir/people/$output_filename" ]
  then
    echo "- Found $output_filename: skipping"
  elif [ -e "$sequence_dir/people/$detector_type" ]
  then
    conda activate smglib
    python evaluate_gta_im_skeleton_sequence.py --batch -s "$sequence_dir" -t "$detector_type" > "$sequence_dir/people/$output_filename"
    echo "- Written metrics to $output_filename"
    conda deactivate
  else
    echo "- Missing skeleton detection results for $detector_type"
  fi
done
