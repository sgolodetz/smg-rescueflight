#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -ne 3 ]
then
  echo "Usage: evaluate_scannet_scene.sh <sequence name> <method tag> <gt method tag>"
  exit 1
fi

# Check that the sequence directory exists.
sequence_dir=`./determine_scannet_sequence_dir.sh "$1"`
if [ -z "$sequence_dir" ]
then
  echo "No such sequence: $1"
  exit 1
fi

# Start the evaluation.
echo "Evaluating $1 ($2)"
recon_dir="$sequence_dir/recon"

# If the evaluation has already been run, avoid running it again.
output_file="c2c_dist-$2-$3.txt"
if [ -f "$recon_dir/$output_file" ]
then
  echo "- Found $output_file: skipping"
  exit 0
fi

# If the reconstruction we want to evaluate is missing, early out.
if [ ! -f "$recon_dir/$2.ply" ]
then
  echo "- Missing $2.ply in $recon_dir"
  exit 1
fi

# If the ground-truth reconstruction is missing, also early out.
if [ ! -f "$recon_dir/$3.ply" ]
then
  echo "- Missing $3.ply in $recon_dir"
  exit 1
fi

# Compute the cloud-to-cloud distances between the two reconstructions using CloudCompare.
echo "- Running CloudCompare..."
"/c/Program Files/CloudCompare/CloudCompare.exe" -SILENT -C_EXPORT_FMT ASC -SEP SPACE -ADD_HEADER -ADD_PTS_COUNT -O "$recon_dir/$2.ply" -O "$recon_dir/$3.ply" -extract_vertices -c2c_dist > /dev/null 2>&1
c2c_file=$(ls "$recon_dir"/*C2C_DIST*.asc)

# Compute the statistics in which we're interested and write them to a file.
echo "- Computing statistics..."

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE\\etc\\profile.d\\conda.sh"
conda activate python2.7

python "$sequence_dir/../../computeStats.py" "$c2c_file" > "$recon_dir/$output_file"
echo "- Writing statistics to: $recon_dir/$output_file"

# Clean up by removing the temporary files output by CloudCompare.
rm "$recon_dir"/*.asc
