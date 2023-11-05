import torch
import numpy as np
import copy
import bisect
from torch.utils.data import DataLoader
from keras_preprocessing.sequence import pad_sequences

def bpe_tokenize(tokenizer, sequence):
    tokens, offsets = [], []
    for token in sequence:
        token_bpe = tokenizer.tokenize(token.text)
        offsets += [(token.begin, token.end) for _ in range(len(token_bpe))]
        tokens += token_bpe
    return tokens, offsets

def nlargest_idx(arr, n):
    ids = np.argpartition(arr, -n)[-n:]
    return ids[np.argsort(-arr[ids])]

def remove_subwords(mask_pos, tokens, offsets):
    if len(mask_pos[1]) > 1:
        to_del = mask_pos[1][1:]
        del tokens[mask_pos[0]][to_del[0] : to_del[-1] + 1]
        del offsets[mask_pos[0]][to_del[0] : to_del[-1] + 1]
        
    mask_pos = (mask_pos[0], mask_pos[1][0])
    return mask_pos, tokens, offsets

def merge(left_o, left_s, right_o, right_s, max_elems):
    res_o, res_s = [], []
    j, i = 0, 0
    while True:
        if len(res_s) == max_elems:
            break
        if i == len(left_s):
            res_o += right_o[j : j + max_elems - len(res_o)]
            res_s += right_s[j : j + max_elems - len(res_s)]
            break
        if j == len(right_s):
            res_o += left_o[i : i + max_elems - len(res_o)]
            res_s += left_s[i : i + max_elems - len(res_s)]
            break
        if left_s[i] > right_s[j]:
            res_o.append(left_o[i])
            res_s.append(left_s[i])
            i += 1
        else:
            res_o.append(right_o[j])
            res_s.append(right_s[j])
            j += 1
    return res_o, res_s

def find_by_offset(offsets, target):
    nums = []
    for sent_num, sent in enumerate(offsets):
        if sent[-1][0] < target[0]:
            continue
        
        for bpe_num, bpe in enumerate(sent):
            if target[0] <= bpe[0] and bpe[1] <= target[1]:
                nums.append(bpe_num)
        return sent_num, nums
    
class MaskedTokenPredictorBERT:
    def __init__(self, model, tokenizer, max_len=250, 
                 device=None, target=0, logits_postprocessor=None, 
                 mean=np.mean, contrast_penalty=0) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.target = target
        self.logits_postprocessor = logits_postprocessor
        self.mean = mean
        self.contrast_penalty = contrast_penalty
    
    def adjust_logits(self, logits, target=0):
        
        if self.logits_postprocessor is not None:
            logits = self.logits_postprocessor(logits, target=target or 0)
            
        return logits
    
    def update_beam(self, 
                    prev_tokens, prev_score, 
                    new_tokens, new_scores, 
                    gen_tokens, gen_scores):
        for i in range(len(gen_scores)):
            final_score = prev_score + gen_scores[i]
            insert_pos = bisect.bisect(new_scores, final_score)
            new_scores.insert(insert_pos, final_score)
            del new_scores[0]
            new_tokens.insert(insert_pos, prev_tokens + [gen_tokens[i]])
            if len(new_tokens) > len(new_scores):
                del new_tokens[0]
                
    def variants(self, bpe_tokens, mask_pos, gen_tokens, gen_scores):
        batch_size = len(bpe_tokens)
        
        if not gen_tokens:
            yield bpe_tokens, [0.0] * batch_size, [[] for _ in range(batch_size)], mask_pos
            return

        for num in range(len(gen_tokens[0])):
            if not gen_tokens[0][num]:
                continue
            
            vars, new_mask, var_tokens, var_scores = [], [], [], []
            for i in range(batch_size):
                new_bpe = copy.deepcopy(bpe_tokens[i])
                
                for seq_num in range(len(gen_tokens[i][num])):
                    new_bpe[mask_pos[i] + seq_num] = gen_tokens[i][num][seq_num]
                
                var_tokens.append(gen_tokens[i][num])
                var_scores.append(gen_scores[i][num])
                new_mask.append(mask_pos[i] + len(gen_tokens[i][num]))
                vars.append(new_bpe)
            yield vars, var_scores, var_tokens, new_mask
        
    def from_tail(self, pred, bpe_tokens, mask_pos, 
                  max_subunits, n_top, target=None):
        res_tokens, res_scores, idx = [], [], 0
        
        while pred[0] == '#' and idx < max_subunits:
            bpe_tokens[mask_pos] = pred
            bpe_tokens.insert(mask_pos, '[MASK]')
            preds, scores = self.predict_unit([bpe_tokens], [mask_pos], n_top=n_top, mask_token=False, target=target)
            pred = preds[0][0]
            res_tokens.append(pred)
            res_scores.append(scores[0][0])
            idx += 1
        return list(reversed(res_tokens)), list(reversed(res_scores))
        
    def generate(self, bpe_tokens_batch, mask_pos_batch, mask_token=True,
                    n_top=5, n_units=1, n_tokens=[1], 
                    fix_multiunit=True, beam_size=10, multiunit_lookup=100, 
                    max_multiunit=10, target=None, **kwargs):
        result_preds, result_scores = [[] for _ in range(len(bpe_tokens_batch))], [[] for _ in range(len(mask_pos_batch))]
        if type(n_tokens) is not list or type(n_tokens) is not tuple:
            n_tokens = list(n_tokens)
        if 1 in n_tokens:
            result_preds, result_scores = self.predict_word(bpe_tokens_batch, mask_pos_batch, mask_token=mask_token, 
                                                            n_top=n_top, n_units=n_units, 
                                                            multiunit_lookup=multiunit_lookup,
                                                            fix_multiunit=fix_multiunit, max_multiunit=max_multiunit, 
                                                            target=target)
        for num in n_tokens:
            if num == 1:
                continue
            pred_tokens, pred_scores = self.predict_sequence(bpe_tokens_batch, mask_pos_batch, 
                                                             n_top=n_top, n_units=n_units, seq_len=num, 
                                                             multiunit_lookup=multiunit_lookup, fix_multiunit=fix_multiunit, 
                                                             max_multiunit=max_multiunit, beam_size=beam_size, target=target)
            for i in range(len(bpe_tokens_batch)):
                result_preds[i], result_scores[i] = merge(result_preds[i], result_scores[i], pred_tokens[i], pred_scores[i], n_top)
        return result_preds, result_scores
        
    def predict_unit(self, bpe_tokens, mask_pos, mask_token, n_top, target=None):
        if target is None:
            target = self.target
            
        bpe_tokens = copy.deepcopy(bpe_tokens)
        max_len = min([max(len(elem) for elem in bpe_tokens) + 2, self.max_len])
        token_ids = []
        
        for i in range(len(bpe_tokens)):
            bpe_tokens[i] = bpe_tokens[i][:max_len - 2]
            
            if mask_token:
                if i >= len(mask_pos) or mask_pos[i] >= len(bpe_tokens[i]):
                    continue
                bpe_tokens[i][mask_pos[i]] = '[MASK]'
            
            bpe_tokens[i] = ['[CLS]'] + bpe_tokens[i] + ['[SEP]']
            token_ids.append(self.tokenizer.convert_tokens_to_ids(bpe_tokens[i]))
        
        token_ids = pad_sequences(token_ids, maxlen=max_len, padding='post', truncating='post', dtype='long')
        attn_mask = torch.tensor(token_ids > 0).long().to(self.device)
        tokens = torch.tensor(token_ids).to(self.device)
        segments = torch.tensor(np.ones_like(token_ids, dtype=int) * target).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            target_sent = self.model(tokens, token_type_ids=segments, attention_mask=attn_mask)[0]
            
            if self.contrast_penalty:
                with torch.no_grad():
                    another = self.model(tokens, token_type_ids=1 - segments, attention_mask=attn_mask)[0]
                diff = torch.softmax(target_sent, dim=-1) - torch.softmax(another, dim=-1) * self.contrast_penalty
                target_sent = torch.log(torch.clamp(diff, min=1e-20))
        
        target_sent = target_sent.detach().cpu().numpy()
        final_scores, final_tokens = [], []
        
        for i in range(target_sent.shape[0]):
            row, idx = target_sent[i], mask_pos[i]
            if idx + 1 >= len(row):
                continue
            logits = self.adjust_logits(row[idx + 1], target=target)
            top_ids = nlargest_idx(logits, n_top)
            top_scores, top_tokens = [target_sent[i][mask_pos[i] + 1][j] for j in top_ids], self.tokenizer.convert_ids_to_tokens(top_ids)
            final_scores.append(top_scores)
            final_tokens.append(top_tokens)
            
        return final_tokens, final_scores
        
    def predict_multiunit(self, bpe_tokens, mask_pos, n_top, n_units, target=None):
        final_tokens, final_scores = [], []
        bpe_tokens = copy.deepcopy(bpe_tokens)
        
        bpe_tokens.insert(mask_pos, '[MASK]')
        preds, scores = self.predict_unit([bpe_tokens], [mask_pos + 1], n_top=n_top, mask_token=False, target=target)
        
        if len(preds) == 0:
            return []
        preds, scores = preds[0], scores[0]
        good_preds = []
        bpe_tokens_batch = []
        for i, pred in (elem for elem in enumerate(preds) if elem[1].startswith('##')):
            temp = copy.deepcopy(bpe_tokens)
            temp[mask_pos + 1] = pred
            bpe_tokens_batch.append(temp)
            good_preds.append((i, pred))
            
        if len(good_preds) == 0:
            return []
        
        loader = DataLoader(bpe_tokens_batch, batch_size=10, collate_fn=lambda _: _)
        preds, sscores = [], []
        for batch in loader:
            new_preds, new_scores = self.predict_unit(batch, [mask_pos for i in range(len(batch))], mask_token=False, n_top=n_top, target=target)
            preds += new_preds
            sscores += new_scores
            
        for i in range(len(preds)):
            result, score = [preds[i][0], good_preds[i][1]], [sscores[i][0], scores[good_preds[i][0]]]
            tail_result, tail_score = self.from_tail(preds[i][0], bpe_tokens_batch[i], mask_pos, max_subunits=n_units - 2, n_top=n_top, target=target)
            result = tail_result + result
            score = tail_score + score
            final_tokens.append(result)
            final_scores.append(score)
        
        return list(zip(final_tokens, final_scores))
                
    def predict_word(self, bpe_tokens, mask_pos, mask_token, 
                     n_top, n_units, fix_multiunit, 
                     multiunit_lookup, max_multiunit, target=None):
        preds, scores = self.predict_unit(bpe_tokens, mask_pos, mask_token=mask_token, n_top=n_top, target=target)
        final_tokens, final_scores = [], []
        for i in range(len(preds)):
            if n_units > 1:
                preds[i], scores[i] = list(reversed(preds[i][:multiunit_lookup])), list(reversed(scores[i][:multiunit_lookup]))
                seqs = self.predict_multiunit(bpe_tokens[i], mask_pos[i], n_top=multiunit_lookup, n_units=n_units, target=target)
                for seq in seqs[:max_multiunit]:
                    seq_pred, seq_score = seq
                    multiunit_token = '_'.join(seq_pred)
                    if fix_multiunit:
                        multiunit_token = multiunit_token.replace('#', '').replace('_', '')
                    multiunit_score = self.mean(seq_score)
                    idx = bisect.bisect(scores[i], multiunit_score)
                    
                    preds[i].insert(idx, multiunit_token)
                    scores[i].insert(idx, multiunit_score)
                preds[i], scores[i] = list(reversed(preds[i])), list(reversed(scores[i]))
                
            final_tokens.append(preds[i][:n_top])
            final_scores.append(scores[i][:n_top])
            
        return final_tokens, final_scores
                
        
    def predict_sequence(self, bpe_tokens, mask_pos, n_top, n_units, 
                         seq_len, multiunit_lookup, fix_multiunit, 
                         max_multiunit, beam_size, target=None):
        bpe_tokens = copy.deepcopy(bpe_tokens)
        batch_size = len(bpe_tokens)
        for i in range(batch_size):
            for num in range(seq_len - 1):
                bpe_tokens[i].insert(mask_pos[i] + 1, '[MASK]')
            
        gen_tokens, gen_scores = [], []
        for num in range(seq_len):
            seq_tokens, seq_scores = [[0. for _ in range(beam_size)] for __ in range(batch_size)], \
                                    [[[] for _ in range(beam_size)] for __ in range(batch_size)]
            for var, var_score, prev_tokens, new_mask in self.variants(bpe_tokens, mask_pos, 
                                                                       gen_tokens, gen_scores):
                top_tokens, top_scores = self.predict_word(var, new_mask, n_top=n_top, mask_token=True, 
                                                           n_units=n_units, fix_multiunit=fix_multiunit,
                                                           multiunit_lookup=multiunit_lookup, 
                                                           max_multiunit=max_multiunit, target=target) 
                for i in range(batch_size):
                    self.update_beam(prev_tokens[i], var_score[i], seq_tokens[i], 
                                     seq_scores[i], top_tokens[i], top_scores[i])
            gen_tokens, gen_scores = seq_tokens, seq_scores
        gen_scores = [[(elem / seq_len) for elem in score] for score in gen_scores]
        return [list(reversed(token)) for token in gen_tokens], [list(reversed(score)) for score in gen_scores]
                    
        