#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Ravi: Use it correctly...: ./scripts/train.sh <config_file> <output_dir>"
fi

if [ -d "$2" ]
then
    rm -rf $2
    echo -e "\e[91mRavi: Removed output dir: $2\e[0m"
fi

echo -e "\e[36mRavi: Using nohup to run in background\e[0m"
echo -e "\e[36mRavi: Run tail -f nohup.out to see progress.\e[0m"
nohup allennlp train $1 -s $2 --include-package drop_library &
