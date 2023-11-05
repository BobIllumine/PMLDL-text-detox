#!/usr/bin/bash
CONDBERT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/interim/condbert_vocab"
if [[ ! -d "${CONDBERT_DIR}" ]]; then
    mkdir -p "${CONDBERT_DIR}"
fi
TRAIN_DIR="${CONDBERT_DIR}/train"
if [[ ! -d "${TRAIN_DIR}" ]]; then
    mkdir -p "${TRAIN_DIR}"
fi
TEST_DIR="${CONDBERT_DIR}/test"
if [[ ! -d "${TEST_DIR}" ]]; then
    mkdir -p "${TEST_DIR}"
fi
TRAIN_NAME_TOXIC="${TRAIN_DIR}/train_toxic"
TRAIN_NAME_NEUTRAL="${TRAIN_DIR}/train_normal"
TEST_NAME_TOXIC="${TEST_DIR}/test_10k_toxic"
TEST_NAME_NEUTRAL="${TEST_DIR}/test_10k_normal"
TOXIC_WORDS="${CONDBERT_DIR}/toxic_words.txt"


if [[ ! -f $TRAIN_NAME_TOXIC ]]; then
    wget "https://raw.githubusercontent.com/s-nlp/detox/main/emnlp2021/data/train/train_toxic" -P "${TRAIN_DIR}"
fi

if [[ ! -f $TRAIN_NAME_NEUTRAL ]]; then
    wget "https://raw.githubusercontent.com/s-nlp/detox/main/emnlp2021/data/train/train_normal" -P "${TRAIN_DIR}"
fi

if [[ ! -f $TEST_NAME_TOXIC ]]; then
    wget "https://raw.githubusercontent.com/s-nlp/detox/main/emnlp2021/data/test/test_10k_toxic" -P "${TEST_DIR}"
fi

if [[ ! -f $TEST_NAME_NEUTRAL ]]; then
    wget "https://raw.githubusercontent.com/s-nlp/detox/main/emnlp2021/data/test/test_10k_normal" -P "${TEST_DIR}"
fi

if [[ ! -f $TOXIC_WORDS ]]; then
    wget "https://raw.githubusercontent.com/s-nlp/detox/main/emnlp2021/style_transfer/condBERT/vocab/toxic_words.txt" -P "${CONDBERT_DIR}"
fi
