#!/bin/sh
# script to copy the headers to all the source files and header files
for f in *.h; do
  if (grep Copyright $f);then
    echo "No need to copy the License Header to $f"
  else
    cat header_file $f > $f.new
    mv $f.new $f
    echo "License Header copied to $f"
  fi
done
