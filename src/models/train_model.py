import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from collections import Counter, defaultdict
import numpy as np
from transformers import BertTokenizer
import pickle
from pathlib import Path

class NgramSalienceCalculator:
    def __init__(self, tox_corpus, norm_corpus):
        ngrams = (1, 1)
        self.vectorizer = CountVectorizer(ngram_range=ngrams)
        tox_matrix = self.vectorizer.fit_transform(tox_corpus)
        self.tox_vocab = self.vectorizer.vocabulary_
        self.tox_count = np.sum(tox_matrix, axis=0)
        
        norm_matrix = self.vectorizer.fit_transform(norm_corpus)
        self.norm_vocab = self.vectorizer.vocabulary_
        self.norm_count = np.sum(norm_matrix, axis=0)
        
    def calculate(self, feature, attr='tox', eps=0.5):
        assert attr in ['tox', 'norm']
        tox_cnt = self.tox_count[0, self.tox_vocab[feature]] if feature in self.tox_vocab else 0.0
        norm_cnt = self.norm_count[0, self.norm_vocab[feature]] if feature in self.norm_vocab else 0.0
        if attr == 'tox':
            return (tox_cnt + eps) / (norm_cnt + eps)
        else:
            return (norm_cnt + eps) / (tox_cnt + eps)
        
def create_corpora(toxic_path, normal_path) -> tuple[list, list]:
    cnt = Counter()
    for filename in [toxic_path, normal_path]:
        with open(filename, 'r') as f:
            for line in f.readlines():
                for token in line.strip().split():
                    cnt[token] += 1
    vocab = {word for word, count in cnt.most_common() if count > 0}
    
    with open(normal_path, 'r') as normal, open(toxic_path, 'r') as toxic:
        tox_corpus = [' '.join([word if word in vocab else '<unk>' for word in line.strip().split()]) for line in toxic.readlines()]
        norm_corpus = [' '.join([word if word in vocab else '<unk>' for word in line.strip().split()]) for line in normal.readlines()]
    return tox_corpus, norm_corpus

def save_words(toxic_corpus, norm_corpus, toxic_path, norm_path):
    calc = NgramSalienceCalculator(toxic_corpus, norm_corpus)
    used_ngrams = set()
    threshold = 4
    with open(norm_path, 'w') as pos_file, open(toxic_path, 'w') as neg_file:
        for gram in set(calc.tox_vocab.keys()).union(set(calc.norm_vocab.keys())):
            if gram in used_ngrams:
                continue
            used_ngrams.add(gram)
            tox_score, norm_score = calc.calculate(gram, attr='tox'), calc.calculate(gram, attr='norm')
            if tox_score > threshold:
                neg_file.writelines(f'{gram}\n')
            elif norm_score > threshold:
                pos_file.writelines(f'{gram}\n')
        
def save_word2coef(toxic_corpus, norm_corpus, out_path, max_iter):
    pipe = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=max_iter))
    X_train, y_train = toxic_corpus + norm_corpus, [1] * len(toxic_corpus) + [0] * len(norm_corpus)
    pipe.fit(X_train, y_train)
    coeffs = pipe[1].coef_[0]
    word2coef = {word: coeffs[idx] for word, idx in pipe[0].vocabulary_.items()}
    with open(out_path, 'wb') as f:
        pickle.dump(word2coef, f)
    
def save_scores(toxic_corpus, norm_corpus, out_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    toxic_cnt, norm_cnt = defaultdict(lambda: 1), defaultdict(lambda: 1)
    for text in toxic_corpus:
        for token in tokenizer.encode(text):
            toxic_cnt[token] += 1

    for text in norm_corpus:
        for token in tokenizer.encode(text):
            norm_cnt[token] += 1
            
    token_toxicities = [toxic_cnt[i] / (norm_cnt[i] + toxic_cnt[i]) for i in range(len(tokenizer.vocab))]
    with open(out_path, 'w') as f:
        for score in token_toxicities:
            f.write(str(score) + '\n')        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--toxic_path', type=str, default=str(Path(__file__).parent.parent.parent / 'data/interim/condbert_vocab/train/train_toxic'))
    parser.add_argument('-n', '--normal_path', type=str, default=str(Path(__file__).parent.parent.parent /'data/interim/condbert_vocab/train/train_normal'))
    parser.add_argument('-o', '--output_path', type=str, default=str(Path(__file__).parent.parent.parent /'data/interim/condbert_vocab/'))
    parser.add_argument('-i', '--max_iter', type=int, default=1000)
    args = parser.parse_args()
    
    tox_corp, norm_corp = create_corpora(args.toxic_path, args.normal_path)
    
    pos_word_path, neg_word_path = args.output_path + 'positive_words.txt', args.output_path + 'negative_words.txt'
    word2coef_path = args.output_path + 'word2coef.pkl'
    score_path = args.output_path + 'token_toxicities.txt'
    
    save_words(tox_corp, norm_corp, pos_word_path, neg_word_path)
    save_word2coef(tox_corp, norm_corp, word2coef_path, args.max_iter)
    save_scores(tox_corp, norm_corp, score_path, score_path)
    
    
    