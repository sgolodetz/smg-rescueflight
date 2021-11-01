#! /bin/bash -e

if [ $# -ne 1 ]
then
  echo "Usage: evaluate_on_scannet.sh <sequence list>"
  exit 1
fi

tr -d '\r' < "$1" | while read f
do
  ./obtain_scannet_sequence.sh "$f"
  ./evaluate_scannet_scenes.sh "$f"
done
