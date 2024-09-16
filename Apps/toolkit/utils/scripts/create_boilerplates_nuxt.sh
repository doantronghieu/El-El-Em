#!/bin/bash

# Array of directories to create
directories=(assets components composables content layouts middleware modules node_modules pages plugins public server utils)


# Loop through directories
for dir in "${directories[@]}"
do
  if [ ! -d "$dir" ]; then
    echo "Creating directory: $dir"
    mkdir "$dir"
  else
    echo "Directory $dir already exists."
  fi
done
