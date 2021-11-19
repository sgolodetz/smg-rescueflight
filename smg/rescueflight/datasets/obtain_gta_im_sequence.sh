#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -ne 1 ]
then
  echo "Usage: obtain_gta_im_sequence.sh <sequence name>"
  exit 1
fi

# Start obtaining the sequence.
echo "Obtaining $1"

# If the sequence directory already exists, there's no need to obtain the sequence, so early out.
sequence_dir=`./determine_sequence_dir.sh gta-im "$1" true`
if [ -e "$sequence_dir" ]
then
  echo "- Already obtained $1: skipping"
  exit 0
fi

# Try to look up the Google Drive data-id for the sequence.
sequence_group=`echo "$1" | perl -pe 's/(.*?)\/.*/\1/g'`
sequence_id=`echo "$1" | perl -pe 's/.*\/(.*)/\1/g'`
data_id=`cat gta_im_data_ids.txt | grep "$sequence_group $sequence_id" | perl -lane 'print $F[2];'`

# If that succeeds:
if [ -n "$data_id" ]
then
  # If the .zip file for the sequence hasn't been downloaded yet, download it.
  root_dir=`./determine_dataset_root_dir.sh gta-im`
  group_folder="$root_dir/$sequence_group"
  zip_filename="$group_folder/$sequence_id.zip"

  if [ ! -f "$zip_filename" ]
  then
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE\\etc\\profile.d\\conda.sh"
    conda activate gdown
    mkdir -p "$group_folder"
    gdown "https://drive.google.com/uc?id=$data_id" -O "$zip_filename"
  fi

  # Unzip the .zip file for the sequence.
  cd "$group_folder"
  unzip "$zip_filename"

# Otherwise:
else
  # Print an error message and exit.
  echo "Error: Could not find Google Drive data-id for $1"
  exit 1
fi
