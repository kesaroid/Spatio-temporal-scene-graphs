from tools.relation_train_net import train
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import vectorize
import pandas as pd
import os
import re

from tqdm.std import tqdm
import utils
from utils import paths_catalog as path
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from collections import Counter
import spacy
import string

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from sklearn.metrics import mean_squared_error

from torch.utils.data import DataLoader
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
import networkx as nx

from models.mac import MACNetwork
from models.mgat import MGAT

from dataloader import AGQA

class Network():
    def __init__(self, ag_video) -> None:
        self.videos = ag_video
        self.instances = self.videos.keys()
        self.batch_size = 1
        self.embed_len = 256

        nlp_text, graph_list, answers = self.preprocess()

        test_len = 1000
        nlp_tr, nlp_te, graph_tr, graph_te, ans_tr, ans_te = model_selection.train_test_split(nlp_text, graph_list, answers, test_size=1000)

        train_ds = AGQA(nlp_tr, graph_tr, ans_tr, self.embed_len)
        test_ds = AGQA(nlp_te, graph_te, ans_te, self.embed_len)
        self.train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True) #, collate_fn=network_collate)
        self.test_dl = DataLoader(test_ds, batch_size=self.batch_size)

        # model = LSTM(0, 128, 64)
        # model =  MACNetwork(vocab_size, 128)
        model = MGAT(nfeat=1, #features.shape[1], 
                nhid=8, 
                nclass=21 + 1, 
                dropout=0.5, 
                nheads=8, 
                alpha=0.2,
                vocab_size = len(self.words)) #nlp_text.shape[1])
        
        if not os.path.exists(path.model_state_dict):
            self.train(model, loss_fn='c_entropy', epochs=10, lr=0.01, step_size=5)
            torch.save(model.state_dict(), path.model_state_dict)
        else:
            model.load_state_dict(torch.load(path.model_state_dict))
        
        loss, accuracy, rmse = self.validation_metrics(model, self.train_dl)
        print('loss: {}, accuracy: {}, rmse loss: {}'.format(loss, accuracy*100, rmse))
        # TODO test

    def preprocess(self, embed=None):        
        if not os.path.exists(path.train_path):
            graph_features = {k: {'features': None, 'adj': None} for k in self.instances}
            video = []; all_Q = []; all_A = []
            for idx in self.instances:
                G = self.videos[idx]['G_gt']
                Q = self.videos[idx]['Questions']['repeat']
                A = self.videos[idx]['Answers']['repeat']
                
                all_Q.extend(Q); all_A.extend(A)
                video.extend([idx for i in range(len(Q))])

                g_embed = self.graph_embedding(idx, G, 'node2vec')
                graph_features[idx]['features'] = g_embed
                graph_features[idx]['adj'] = nx.linalg.graphmatrix.adjacency_matrix(G).todense()
                # all_G.extend([g_embed for i in range(len(Q))])
                
            # assert len(all_G) == len(all_Q)
            self.df = pd.DataFrame(list(zip(video, all_Q, all_A)), columns=['id', 'question', 'answer'])
            utils.save_pkl(self.df, path.train_path)
            utils.save_pkl(graph_features, path.gfeat_path)
        else:
            self.df = utils.load_pkl(path.train_path)
            self.gfeat = utils.load_pkl(path.gfeat_path)
        
        X_nlp = self.text_embedding()
        self.labels = len(np.unique(self.df['answer']))
        print("Unique answers: ", self.labels)

        # self.df.to_csv('finalRun/corpus.csv')
        # X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.2) # TODO

        # X_nlp = list(zip(self.df['encoded'], self.df['q_length']))
        X_nlp = list(self.df['encoded'])

        # X_g = [(self.gfeat[idx]['features'], self.gfeat[idx]['adj']) for idx in self.df['id']]
        X_g = [self.videos[idx]['G_gt'] for idx in self.df['id']]
        y = list(self.df['answer'])

        assert len(X_g) == len(X_nlp)

        return X_nlp, X_g, y

    def text_embedding(self,):
        tok = spacy.load('en_core_web_sm')
        def tokenize(text):
            text = re.sub(r'[^\w\s]?', '', str(text).lower().strip())
            return [token.text for token in tok.tokenizer(text)]
        
        def encode_sentence(text, vocab2index, N=10):
            tokenized = tokenize(text)
            encoded = np.zeros(N, dtype=int)
            enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
            length = min(N, len(enc1))
            encoded[:length] = enc1[:length] # TODO Length not needed?
            return encoded, length

        ## TEEXT EMB #2
        counts = Counter()
        for index, row in self.df.iterrows():
            counts.update(tokenize(row['question']))

        print(counts)
        # Delete repeating words
        for word in list(counts):
            if counts[word] > 5000:
                del counts[word]

        #creating vocabulary
        vocab2index = {"":0, "UNK":1}
        self.words = ["", "UNK"]
        for word in counts:
            vocab2index[word] = len(self.words)
            self.words.append(word)

        # texts = []
        # for index, row in self.df.iterrows():
        #     texts.append(tokenize(row['question']))

        # texts = [" ".join(t) for t in texts]
        # vectorizer = TfidfVectorizer(use_idf=True)
        # x = vectorizer.fit_transform(texts).toarray()
        # return x

        self.df['encoded'] = self.df['question'].apply(lambda x: np.array(encode_sentence(x,vocab2index)))

    def graph_embedding(self, idx, G, mode=None, save=False):
        def node2vec(G):
            show_tsne=False
            node2vec = Node2Vec(G, dimensions=128, walk_length=5, num_walks=100, workers=4)
            model = node2vec.fit(window=10, min_count=1, batch_words=4) 
            node_ids = model.wv.index_to_key  # list of node IDs
            node_embeddings = (
                model.wv.vectors
            )  # numpy.ndarray of size number of nodes times embeddings dimensionality
            # if save:
            #     gmodel_path = os.path.join(path.g_embed_path, idx.split('.')[0] + '.emb')
            #     if not os.path.exists(gmodel_path):
            #         os.makedirs(gmodel_path)
            #     model.save(gmodel_path)

            if show_tsne:
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2)
                node_embeddings_2d = tsne.fit_transform(node_embeddings)
                # draw the points
                alpha = 0.7
                # label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
                # node_colours = [label_map[target] for target in node_targets]

                plt.figure(figsize=(10, 8))
                plt.scatter(
                    node_embeddings_2d[:, 0],
                    node_embeddings_2d[:, 1],
                    cmap="jet",
                    alpha=alpha,
                )
                plt.show()
            return node_embeddings
            # return node_embeddings

        if mode == 'adj':
            return nx.adjacency_matrix(G)
        elif mode == 'node2vec':
            return node2vec(G)     
        else:
            return None           

    def train(self, model, loss_fn='nll', epochs=10, lr=0.01, step_size=5):
        # lstm = LSTM(input_size=32, hidden_layer_size=64, output_size=64)
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
        loss_func = nn.CrossEntropyLoss()
        
        for i in range(epochs):
            model.train().to('cuda')
            sum_loss = 0.0
            total = 0
            for ntext, ngraph, nadj, y in tqdm(self.train_dl):
                ntext, ngraph, nadj = (
                            ntext.long().to('cuda'),
                            ngraph[0].float().to('cuda'),
                            nadj[0].to('cuda')
                )
                y = y.long().to('cuda')

                optimizer.zero_grad()
                y_pred = model(ntext, ngraph, nadj)
                # print(y_pred)
                # nll = -F.log_softmax(y_pred, dim=0)                          # 2 
                # loss = (nll * y / 10).sum(dim=0).mean()
                # loss = F.nll_loss(y_pred, y)              # 3
                loss = loss_func(y_pred, y)

                loss.backward()
                optimizer.step()
                sum_loss += loss.item()*y.shape[0]
                total += y.shape[0]
            
            scheduler.step()
            # val_loss, val_acc, val_rmse = self.validation_metrics(model, val_dl)
            # print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))
            print("Train: epoch %i loss %.3f, lr %.5f" % (i+1, sum_loss/total, optimizer.param_groups[0]['lr']))
    
    def validation_metrics(self, model, valid_dl):
        
        # with torch.no_grad():
        model.eval().to('cuda')
        correct = 0
        total = 0
        sum_loss = 0.0
        sum_rmse = 0.0
        for ntext, ngraph, nadj, y in valid_dl:
            ntext = ntext.long().to('cuda')
            ngraph = ngraph[0].float().to('cuda')
            nadj = nadj[0].to('cuda')
            y = y.to('cuda')

            y_pred = model(ntext, ngraph, nadj)
            loss = F.cross_entropy(y_pred, y)
            pred = torch.max(y_pred, 1)[1]
            correct += (pred == y).float().sum()
            total += y.shape[0]
            sum_loss += loss.item()*y.shape[0]
            sum_rmse += np.sqrt(mean_squared_error(pred.cpu(), y.cpu().unsqueeze(-1)))*y.shape[0]
        return sum_loss/total, correct/total, sum_rmse/total
