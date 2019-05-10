#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Blarry: Use it correctly...: ./scripts/pred_json.sh <archive_file> <input_file> <output_file>"
fi

allennlp predict $1 $2 \
  --output-file $3 \
  --predictor naqanet \
  --include-package drop_library \
  --use-dataset-reader \
  --overrides "{dataset_reader: {type: 'drop', token_indexers: { tokens: { type: 'single_id', lowercase_tokens: true }, token_characters: { type: 'characters', min_padding_length: 5 } }, passage_length_limit: 1000, question_length_limit: 100, skip_when_all_empty: [], instance_format: 'drop'}}" \
  --silent
