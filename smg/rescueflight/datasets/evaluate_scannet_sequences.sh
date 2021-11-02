#! /bin/bash -e

if [ $# -ne 1 ]
then
  echo "Usage: evaluate_scannet_sequences.sh <sequence list>"
  exit 1
fi

tr -d '\r' < "$1" | while read f
do
  ./evaluate_scannet_sequence.sh "$f"
done

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE\\etc\\profile.d\\conda.sh"
conda activate smglib

python make_scannet_tables.py -s "$1"
