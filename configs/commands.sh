#!/bin/bash

#SBATCH --exclusive

# Example file for a batch script to be encrypted
# Replace all placeholders in angle brackets (including the brackets)
# by the actual values
#
# <input data>  is the name of the input data container. E.g. "input_data" referring to the file "input_data.img"
# <output data> is similar but for the result data
# <sif name>    is the name of the key for the Singularity image (*.sif) which usually is just the filename, e.g. "image"
# <... token>   are the tokens corresponding to that secret

getkey 'input_data' hvs.CAESI
getkey 'output_data' hvs.CAESI
getkey 'image' hvs.CAESI

mount_data "input_data" || { echo "Failed mounting input data"; exit 1; }
trap "umount_data input_data" EXIT
mount_data "output_data" || { echo "Failed mounting output data"; exit 1; }
trap "umount_data input_data; umount_data output_data" EXIT

# The program itself will run within the Singularity image. You need to point
# the program to the input and output folders and add the program-specific
# arguments. Both can be incorporated via arguments or configuration files.
singularity exec --pem-path=/keys/image.key \
  --contain \
  --writable-tmpfs \
  --bind /mnt/input_data:/input:ro \
  --bind /mnt/output_data:/output:rw \
  ~/image.sif \
  python /app/src/train.py --input-dir /input/HAM10000 --output-dir /output --config /app/configs/realistic.yaml





