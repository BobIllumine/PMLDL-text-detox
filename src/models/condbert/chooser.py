from typing import Any
import numpy as np
from flair.data import Sentence
from flair.embeddings import WordEmbeddings


def cosine(v1, v2):
    return np.dot(v1, v2) / np.sqrt(sum(v1 ** 2) * sum(v2 ** 2) + 1e-10)

class SimilarityChooser:
    def __init__(self, coeff=100, tokenizer=None):
        self.coeff = coeff
        self.tokenizer = tokenizer
        self.embedding = WordEmbeddings('glove')
    
    def embed(self, text):
        tokens = self.embedding.embed(Sentence(text))[0]
        return np.mean([t.embedding.cpu().numpy() for t in tokens], axis=0) if tokens else np.zeros(self.embedding.embedding_length)
    
    def decode(self, tokens):
        if isinstance(tokens, str):
            return tokens
        if self.tokenizer:
            return self.tokenizer.convert_tokens_to_string(tokens)
        return ' '.join(tokens).replace(' ##', '')
    
    def __call__(self, hypotheses, original=None, scores=None):
        embedded = self.embed(self.decode(original))
        nominees = [
            (hypothesis, score, cosine(embedded, self.embed(self.decode(hypothesis)))) for hypothesis, score in zip(hypotheses, scores)
        ]
        nominees = sorted(nominees, key=lambda x: x[1] + x[2] * self.coeff, reverse=True)
        return nominees[0][0]