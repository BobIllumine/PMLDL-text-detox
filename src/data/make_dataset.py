import subprocess
import argparse
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
from pathlib import Path

def run_scripts(script_list):
    for script in script_list:
        print(f'Running {script}')
        subprocess.call(['bash', script])
        
def combine(path, train_normal, train_toxic, neutral_ratio, toxic_ratio):
    paranmt_df = pd.read_csv(path, sep="\t", index_col=0)
    tokenizer = WordPunctTokenizer()
    toxic_ref = list(map(lambda x: f'{" ".join(tokenizer.tokenize(x))}\n', paranmt_df[paranmt_df.ref_tox >= toxic_ratio]['reference'].tolist()))
    with open(train_toxic, 'a') as f:
        f.writelines(toxic_ref)
        
    neutral_ref = list(map(lambda x: f'{" ".join(tokenizer.tokenize(x))}\n', paranmt_df[paranmt_df.ref_tox <= neutral_ratio]['reference'].tolist()))
    with open(train_normal, 'a') as f:
        f.writelines(neutral_ref)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clear_vocab', action='store_true', default=False)
    parser.add_argument('--combine', action='store_true', default=True)
    parser.add_argument('--neutral_ratio', type=float, default=0.2)
    parser.add_argument('--toxic_ratio', type=float, default=0.8)
    args = parser.parse_args()
    
    scripts = [str(Path(__file__).parent.parent.parent / 'data/clean_vocab.sh')] if args.clear_vocab else []
    scripts += [str(Path(__file__).parent.parent.parent / 'data/download_jigsaw.sh'), 
                str(Path(__file__).parent.parent.parent / 'data/download_paradetox.sh'), 
                str(Path(__file__).parent.parent.parent / 'data/download_paranmt.sh')]
    run_scripts(scripts)
    
    paranmt_path, train_normal_path, train_toxic_path = str(Path(__file__).parent.parent.parent / 'data/raw/filtered.tsv'), \
                                                        str(Path(__file__).parent.parent.parent / 'data/interim/train_normal'), \
                                                        str(Path(__file__).parent.parent.parent / 'data/interim/train_toxic')
    
    if args.combine:
        combine(paranmt_path, train_normal_path, train_toxic_path, args.neutral_ratio, args.toxic_ratio)
    
    
    