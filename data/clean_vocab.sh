#!/usr/bin/bash
CONDBERT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/interim/condbert_vocab"
if [[ -d "${CONDBERT_DIR}" ]]; then
    if [[ -d "${CONDBERT_DIR}/train" ]]; then
        find "${CONDBERT_DIR}/train" -type f -delete
    fi
    if [[ -d "${CONDBERT_DIR}/test" ]]; then
        find "${CONDBERT_DIR}/test" -type f -delete
    fi
    find "${CONDBERT_DIR}" -type f -delete
fi
