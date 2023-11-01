#!/usr/bin/bash
RAW_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/raw"
ARCHIVE_NAME="${RAW_DIR}/filtered_paranmt.zip"
FILE_NAME="${RAW_DIR}/filtered.tsv"



if [[ ! -f $ARCHIVE_NAME ]]; then
    wget "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip" -P "${RAW_DIR}"
fi

if [[ ! -f $FILE_NAME ]]; then
    unzip "${ARCHIVE_NAME}" -d "${RAW_DIR}"
fi