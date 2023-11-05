import torch
from collections import defaultdict

def group_by_first(texts, tokenizer) -> defaultdict:
    grouped = defaultdict(list)
    sequences = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    for seq in sequences:
        grouped[seq[0]].append(seq)
    return grouped

def default_chooser(hypotheses, original=None, **kwargs):
    return hypotheses[0]

def log_likelihood(inputs, results):
    probs = torch.log_softmax(results.logits, dim=-1)
    seq = inputs['input_ids']
    prob = torch.gather(probs, 2, seq[:, :, None]).squeeze(-1)
    return prob * inputs['attention_mask']

class CondBERTRewriter:
    def __init__(self, model, tokenizer, device, 
                 neg_words, pos_words, word2coef, 
                 tox_scores, predictor=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.neg_words = group_by_first(neg_words, self.tokenizer)
        self.pos_words = group_by_first(pos_words, self.tokenizer)
        self.word2coef = word2coef
        self.tox_scores = torch.tensor(tox_scores).to(self.device)
        self.predictor = predictor
        
        self.vocab = {v: k for k, v in tokenizer.vocab.items()}
        self.mask_idx = self.tokenizer.convert_tokens_to_ids('[MASK]')
        
    def merge_subtokens(self, token_ids):
        idx = []
        for i, token_id in enumerate(token_ids):
            text = self.vocab[token_id]
            if text.startswith('##'):
                idx.append(i)
            else:
                if idx:
                    tokens = [self.vocab[token_ids[x]] for x in idx]
                    word = ' '.join([tokens[0]] + [t[2:] for t in tokens[1:]])
                    yield idx, word
                idx = [i]
    
    def fast_mask(self, tokens: str, bad_words=None, min_bad_score=0, 
                  aggressive=True, max_bad_score=0.5, target=0):
        if bad_words is None:
            bad_words = self.neg_words if target == 0 else self.pos_words
            
        sentences = [self.tokenizer.encode(tokens, add_special_tokens=True)]
        sentences_torch = torch.tensor(sentences).to(self.device)
        masks = torch.zeros_like(sentences_torch)
        
        for sent_id, sent in enumerate(sentences):
            for token_id, token in enumerate(sent):
                for hypothesis in bad_words.get(token, []):
                    n = len(hypothesis)
                    if sent[token_id : token_id + n] == hypothesis:
                        for step in range(n):
                            masks[sent_id, token_id + step] = 1
                        for offset, next in enumerate(sent[token_id + n : ]):
                            if self.tokenizer.convert_ids_to_tokens(next).startswith('##'):
                                masks[sent_id, token_id + n + offset] = 1
                            else:
                                break
            if sum(masks[sent_id].cpu().numpy()) == 0 or aggressive:
                scored = []
                for idx, word in self.merge_subtokens(sent):
                    score = self.word2coef.get(word, 0) * (1 - 2 * target)
                    if score:
                        scored.append([idx, word, score])
                if scored:
                    max_score = max(s[2] for s in scored)
                    if max_score > min_bad_score:
                        for idx, word, score in scored:
                            if score >= max(min_bad_score, max_score * max_bad_score):
                                masks[sent_id, idx] = 1
                                
        return sentences_torch, masks
    
    def convert_mask(self, token_ids, mask_ids, duplicate=False, start_from=0):
        temp_tokens = [self.tokenizer.convert_ids_to_tokens(token_ids[0])[1:-1]]
        mask_pos = None
        tokens, mask_tokens, masked = [], [], False
        
        for i, is_masked in enumerate(mask_ids[0][1:-1]):
            token = temp_tokens[0][i]
            
            if not masked:
                if is_masked and i >= start_from and not token.startswith('##'):
                    masked = True
                    mask_pos = [i]
                    mask_tokens.append(token)
                tokens.append(token)
            else:
                if not is_masked or not token.startswith('##'):
                    tokens.extend(temp_tokens[0][i:])
                    break
                else:
                    mask_tokens.append(token)
                    
        tokens = [tokens]
        if duplicate:
            tokens = [temp_tokens[0] + ['[SEP]'] + tokens[0]]
            mask_pos[0] += len(temp_tokens[0]) + 1
        
        return tokens, mask_pos, mask_tokens
        
    def replace(self, text, span_detector=None, predictor=None, verbose=True, 
                chooser=default_chooser, n_tokens=(1, 2, 3), n_top=10, 
                mask_token=False, max_steps=1000, target=0, 
                **pred_args):
        if span_detector is None:
            span_detector = self.fast_mask
        if predictor is None:
            predictor = self.predictor
        generated_text = text
        first_idx = 0
        for i in range(max_steps):
            tokens_idx, mask_idx = span_detector(generated_text, target=target)
            if sum(mask_idx[0][first_idx + 1:]) == 0:
                break
            tokens, mask_pos, mask_tokens = self.convert_mask(tokens_idx, mask_idx, duplicate=False, start_from=first_idx)
            
            if mask_pos is None:
                return generated_text
            texts, scores = predictor.generate(tokens, mask_pos, 
                                               n_tokens=list(n_tokens), n_top=n_top, 
                                               fix_multiunit=True, mask_token=mask_token, 
                                               target=target, **pred_args)
            prev_replacement = chooser(hypotheses=texts[0], scores=scores[0], original=mask_tokens)
            if isinstance(prev_replacement, str):
                prev_replacement = [prev_replacement]
            replacement = [t for w in prev_replacement for t in w.split('_')]
            if verbose:
                print(mask_tokens, '->', replacement)
            generated_tokens = tokens[0][:mask_pos[0]] + replacement + tokens[0][mask_pos[0] + 1:]
            generated_text = self.tokenizer.convert_tokens_to_string(generated_tokens)
            first_idx = mask_pos[0] + len(prev_replacement)
            
        return generated_text