import argparse
from transformers import BertTokenizer, BertForMaskedLM
from condbert.chooser import SimilarityChooser
from condbert.predictor import MaskedTokenPredictorBERT
from condbert.rewriter import CondBERTRewriter
import pickle
import string
import numpy as np
import torch
from pathlib import Path
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--n_tokens', type=tuple, default=(1, 2, 3))
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--n_units', type=int, default=5)
    parser.add_argument('--sim_coeff', type=int, default=50)
    parser.add_argument('--vocab_path', type=str, default=str(Path(__file__).parent.parent.parent / 'data/interim/condbert_vocab'))
    parser.add_argument('--penalty', type=float, default=7)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForMaskedLM.from_pretrained(args.model_name)
    model.to(device)
    negative_words, positive_words = [], []
    with open(f'{args.vocab_path}/negative_words.txt', 'r') as f:
        tmp = f.readlines()
    negative_words = list(map(lambda x: x[:-1], tmp))
    with open(f'{args.vocab_path}/toxic_words.txt', 'r') as f:
        tmp = f.readlines()
    negative_words += list(map(lambda x: x[:-1], tmp))
    with open(f'{args.vocab_path}/positive_words.txt', 'r') as f:
        tmp = f.readlines()
    positive_words = list(map(lambda x: x[:-1], tmp))
    with open(f'{args.vocab_path}/word2coef.pkl', 'rb') as f:
        word2coef = pickle.load(f)
        
    token_toxicities = []
    with open(f'{args.vocab_path}/token_toxicities.txt', 'r') as f:
        for line in f.readlines():
            token_toxicities.append(float(line))
    token_toxicities = np.array(token_toxicities)
    token_toxicities = np.maximum(0, np.log(1 / (1 / token_toxicities - 1)))

    for token in string.punctuation:
        token_toxicities[tokenizer.encode(token)][1] = 3
    token_toxicities[tokenizer.encode('you')][1] = 0
    predictor = MaskedTokenPredictorBERT(model, tokenizer, max_len=250, target=args.target, device=device)
    chooser = SimilarityChooser(coeff=args.sim_coeff, tokenizer=tokenizer)
    rewriter = CondBERTRewriter(model, tokenizer, device, 
                                negative_words, positive_words, 
                                word2coef, token_toxicities, predictor)
    def adjust_logits(logits, target=0):
        return logits - rewriter.tox_scores.cpu().numpy() * (1 - 2 * target) * args.penalty
    
    predictor.logits_postprocessor = adjust_logits
    os.makedirs(Path(args.output_path).parent, exist_ok=True)
    with open(args.input_path, 'r') as f, open(args.output_path, 'w') as out:
        lines = [line.strip() for line in f.readlines()]
        for line in tqdm(lines):
            out.write(f'{rewriter.replace(line, chooser=chooser, n_tokens=args.n_tokens, n_units=args.n_units, target=args.target, verbose=False)}\n')