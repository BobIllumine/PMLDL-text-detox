# Practical Machine Learning and Deep Learning - Assignment 1 - Text De-toxification

## Task description

Text Detoxification Task is a process of transforming the text with toxic style into the text with the same meaning but with neutral style.

> Formal definition of text detoxification task can be found in [Text Detoxification using Large Pre-trained Neural Models by Dale et al., page 14](https://arxiv.org/abs/2109.08914)

Your assignment is to create a solution for detoxing text with high level of toxicity. It can be a model or set of models, or any algorithm that would work. 

## How to run

In order to run the model, first you need to import the dataset. It can be easily done with the following command:
```bash
python ./src/data/make_dataset.py --combine
```

Next, the vocabulary for the model needs to be created:
```bash
python ./src/models/train_model.pypython ./src/data/train_model.py
```

And, finally, the inference:
```bash
python ./src/models/predict_model.py --input_path /path/to/input --output_path /path/to/output
```