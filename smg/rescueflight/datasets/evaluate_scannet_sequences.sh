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
