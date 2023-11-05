#!/usr/bin/bash
EXTERNAL_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/external"
if [[ ! -d "${EXTERNAL_DIR}" ]]; then
    mkdir -p "${EXTERNAL_DIR}"
fi
FILE_NAME="${EXTERNAL_DIR}/paradetox.tsv"


if [[ ! -f $FILE_NAME ]]; then
    wget "https://raw.githubusercontent.com/s-nlp/paradetox/main/paradetox/paradetox.tsv" -P "${EXTERNAL_DIR}"
fi
