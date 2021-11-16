#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -ne 2 ]
then
  echo "Usage: obtain_gta_im_sequence.sh <sequence group> <sequence name>"
  exit 1
fi

# Try to look up the Google Drive data-id for the sequence.
data_id=`cat gta_im_data_ids.txt | grep "$1 $2" | perl -lane 'print $F[2];'`

# If that succeeds:
if [ -n "$data_id" ]
then
  # Download the .zip file for the sequence.
  CONDA_BASE=$(conda info --base)
  source "$CONDA_BASE\\etc\\profile.d\\conda.sh"
  conda activate gdown
  zip_folder="C:/foo/$1"  # FIXME
  mkdir -p "$zip_folder"
  zip_filename="$zip_folder/$2.zip"
  gdown "https://drive.google.com/uc?id=$data_id" -O "$zip_filename"
  cd "$zip_folder"
  unzip "$zip_filename"

# Otherwise:
else
  # Print an error message and exit.
  echo "Error: Could not find Google Drive data-id for $1/$2"
  exit 1
fi
