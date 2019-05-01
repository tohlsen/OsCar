#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Blarry: Use it correctly...: ./scripts/pred_txt.sh <archive_file> <input_file> <output_file>"
fi

allennlp predict $1 $2 \
  --output-file $3 \
  --predictor naqanet \
  --include-package drop_library \
  --use-dataset-reader \
  --silent
