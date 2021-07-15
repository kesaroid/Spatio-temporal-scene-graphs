import os
from re import X
import types
import torch
import json
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

import utils
from utils.dataset import AGDataset
from utils.qa import QA, Text_classification
from utils.graph import create_graph
from utils import paths_catalog as path

from model import Network

def main():
    ag = AGDataset(cache=True)
    draw = False
    train_model = True

    if not os.path.exists(path.corpus_path):
        ag.create_db()
        for video in tqdm(ag.instances):
            # print(video)
            G = create_graph(ag.videos[video], ag.cat2pred)
            ag.videos[video]['G_gt'] = G
            if draw:
                nx.draw(G, with_labels=True, font_weight='bold')
                nx.random_layout(G)
                plt.show()
            
            Q = QA(ag.videos[video], G, ag.cat2pred)
            ag.videos[video]['Questions'], ag.videos[video]['Answers'] = Q.create_questions()

        utils.save_pkl(ag.videos, path.corpus_path)
    
    else:
        print('Loading from cache..')
        ag.videos = utils.load_pkl(path.corpus_path)

#     identifier = Text_classification(ag.videos)
#     identifier.train()
    model = Network(ag.videos)


if __name__ == "__main__":
    main()