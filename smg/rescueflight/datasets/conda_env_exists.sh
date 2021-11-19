#! /bin/bash -e

# Check that the script is being used correctly.
if [ $# -ne 1 ]
then
  echo "Usage: conda_env_exists.sh <environment name>"
  exit 1
fi

# Enable the conda command.
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE\\etc\\profile.d\\conda.sh"

# Check whether the specified environment exists. Print 1 if yes, and 0 if no.
if [ -z `conda info --envs | tail -n +3 | perl -lane 'print $F[0]'  | grep "^$1$" || true` ]
then
  echo "0"
else
  echo "1"
fi
