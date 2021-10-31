#! /bin/bash -e

if [ $# -ne 3 ]
then
  echo "Usage: evaluate_scannet_scene.sh <sequence name> <tag> <gt tag>"
  exit 1
fi

sequence_dir=`./determine_scannet_sequence_dir.sh "$1"`
if [ -z "$sequence_dir" ]
then
  echo "No such sequence: $1"
  exit 1
fi

recon_dir="$sequence_dir/recon"

if [ ! -f "$recon_dir/$2.ply" ]
then
  echo "Missing $2.ply in $recon_dir"
  exit 1
fi

if [ ! -f "$recon_dir/$3.ply" ]
then
  echo "Missing $3.ply in $recon_dir"
  exit 1
fi

"/c/Program Files/CloudCompare/CloudCompare.exe" -SILENT -C_EXPORT_FMT ASC -SEP SPACE -ADD_HEADER -ADD_PTS_COUNT -O "$recon_dir/$2.ply" -O "$recon_dir/$3.ply" -extract_vertices -c2c_dist > /dev/null 2>&1
c2c_file=$(ls "$recon_dir"/*C2C_DIST*.asc)

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE\\etc\\profile.d\\conda.sh"
conda activate python2.7

python /c/datasets/computeStats.py "$c2c_file" > "$recon_dir/c2c_dist-$2-$3.txt"
rm "$recon_dir"/*.asc
