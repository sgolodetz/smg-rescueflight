#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -ne 1 ]
then
  echo "Usage: evaluate_scannet_sequences.sh <sequence list file>"
  exit 1
fi

# Read a list of ScanNet sequences from a file, and run the evaluation for each one in turn.
tr -d '\r' < "$1" | while read sequence_name
do
  ./evaluate_scannet_sequence.sh "$sequence_name"
done

# Make the output tables for the paper.
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE\\etc\\profile.d\\conda.sh"
conda activate smglib

python make_scannet_tables.py -s "$1"
