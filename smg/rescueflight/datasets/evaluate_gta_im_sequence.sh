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

# Repeatedly reconstruct the sequence using the various different methods we want to compare.
# TODO: Improve this comment (include something about skeletons and people masks).
./reconstruct_gta_im_sequence.sh "$1"

# Evaluate the reconstructions for each of the different methods in turn.
#for method_tag in dvmvs_4m_gt dvmvs_pp_4m_gt mvdepth_4m_gt mvdepth_pp_4m_gt
#do
#  ./evaluate_gta_im_scene.sh "$1" "$method_tag" gt_gt
#done

# TODO: Comment here, plus add the other methods.
for method_tag in gt
do
  echo "Evaluating people masks for $1 ($method_tag)"
  output_filename="people_mask_metrics-$method_tag.txt"
  if [ -e "$sequence_dir/people/$output_filename" ]
  then
    echo "- Found $output_filename: skipping"
  else
    python evaluate_gta_im_people_mask_sequence.py -s "$sequence_dir" -t "$method_tag" > "$sequence_dir/people/$output_filename"
    echo "- Written metrics to $output_filename"
  fi
done

# TODO: Comment here, plus add the other methods.
for method_tag in gt
do
  python evaluate_gta_im_skeleton_sequence.py --batch -s "$sequence_dir" -t "$method_tag"
done
