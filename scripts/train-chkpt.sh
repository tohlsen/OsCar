#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Ravi: Use it correctly...: ./scripts/train.sh <config_file>"
fi

expName=$(echo $1 | cut -f2 -d/)
expName=$(echo $expName | cut -f1 -d.)
outDir="out/$expName"
nohupName="$expName.out"

echo -e "\e[36mRavi: Using nohup to run in background\e[0m"
echo -e "\e[36mRavi: Run tail -f $nohupName to see progress.\e[0m"
echo $outDir
nohup allennlp train $1 -s $outDir -r --include-package drop_library > $nohupName 2>&1 &
