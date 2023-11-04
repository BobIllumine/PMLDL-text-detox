import torch
import numpy as np
import copy
import bisect
from torch.utils.data import DataLoader
from keras.preprocessing.sequence import pad_sequences

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
        if len(res_o) >= max_elems:
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
    def __init__(self) -> None:
        pass