import argparse
import collections
import glob
import logging
import os
import sys

sys.path.append(os.getcwd())
import jieba
import numpy as np
from elmoformanylangs import Embedder

logging.basicConfig(level=logging.ERROR)

def main(data_dir:str, out_dir:str, ssds_threshold:int):
    """ Extract feats for audio.
    
    Args:
        data_dir (str): Directory of EATD-Corpus
        out_dir (str): Directory to save .npz file
        ssds_threshold (int): Threshold for standard SDS score
                              正常     :       ssds < 50
                              轻度     : 50 <= ssds < 59
                              中度至重度: 60 <= ssds < 69
                              重度     : 70 <= ssds

    Returns:
        None
    """
    elmo = Embedder(os.path.join(os.getcwd(), 'models/zhs.model'))
    
    feats = []
    targets = []

    for item_dir in sorted(glob.glob(data_dir + r'/*/*'), 
                        reverse = False, 
                        key = lambda x: int(x.split('_')[-1])):
        answer_dict = collections.defaultdict(lambda: None)
        for item_path in glob.glob(item_dir + r'/*.txt'.format(item_dir)):
            if 'label' in item_path:
                continue
            with open(item_path, 'r') as file:
                answer = file.readlines()[0]
            answer_sents_iter = jieba.cut(answer, cut_all=False)
            answer_sents = [seg for seg in answer_sents_iter]
            
            # Sentence to Elmo feat
            item_type = item_path.split('/')[-1].replace('.txt', '')
            answer_dict[item_type] = answer_sents
            
        answers = [answer_dict[k] for k in ['positive', 'neutral', 'negative']]
        feats.append([np.array(item).mean(axis=0) for item in elmo.sents2elmo(answers)])
            
        # 2. Load standard SDS score as target
        with open(item_dir + r'/new_label.txt') as file:
            target = float(file.readline())
        targets.append(target)
        
    feats = np.array(feats)
    targets = np.array(targets)
    dep_idxs = np.where(targets >= ssds_threshold)[0]
    non_idxs = np.where(targets < ssds_threshold)[0]
    print("Number of samples: {} ( dep-{}, non dep-{} )".format(feats.shape[0], len(dep_idxs), len(non_idxs)))
    np.savez(os.path.join(out_dir, 'text_feats.npz'), feats=feats, targets=targets, dep_idxs=dep_idxs, non_idxs=non_idxs)
    print("text_feats.npz has been saved in dir: {}".format(out_dir))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_dir', type=str, help='Directory of EATD-Corpus', required=True)
    parser.add_argument('-o', '--out_dir', type=str, help='Directory to save text_feats.npz file', default=None)
    parser.add_argument('-t', '--ssds_threshold', type=int, help='Threshold for standard SDS score', default=53)
    
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = os.path.join(os.getcwd(), 'data/EATD-Feats')
    opts = vars(args)
    main(**opts)
