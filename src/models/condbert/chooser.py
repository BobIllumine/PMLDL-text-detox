from typing import Any
import numpy as np
from flair.data import Sentence
from flair.embeddings import WordEmbeddings


def cosine(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

class SimilarityChooser:
    def __init__(self, coeff=100, tokenizer=None):
        self.coeff = coeff
        self.tokenizer = tokenizer
        self.embedding = WordEmbeddings('glove')
    
    def embed(self, text):
        tokens = self.embedding.embed(Sentence(text))[0]
        return tokens if tokens is not None else np.zeros(self.embedding.embedding_length)
    
    def decode(self, tokens):
        if isinstance(tokens, str):
            return tokens
        if self.tokenizer:
            return self.tokenizer.convert_tokens_to_string(tokens)
        return ' '.join(tokens).replace(' ##', '')
    
    def __call__(self, hypotheses, original=None, scores=None):
        embedded = self.embed(self.decode(original))
        nominees = sorted([
            (hypothesis, score, cosine(embedded, self.embed(self.decode(hypothesis)))) for hypothesis, score in zip(hypotheses, scores)
        ], key=lambda x: x[1] + x[2] * self.coeff, reverse=True)
        return nominees[0][0]