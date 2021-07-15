import networkx as nx
import more_itertools as mit


########################################################

def create_graph(video, cat2pred, mode='groundtruth'):

        G = nx.DiGraph()
        
        G.add_nodes_from(video['objs'])
        for i in range(len(video['frames'])):
            frame = video['frames'][i].split('.')[0]
            G.add_node(i, timestepframe=video['frames'][i], frame=frame) # i or frame
            G.add_edge('person', i) # i or frame
            for triplet in video[mode][i]:
                G.add_edge(i, triplet[1]) # i or frame
                G.add_edge(triplet[1], triplet[-1])
        

        # print("Nodes: ", G.nodes(data=True)) # Show all nodes
        # print("Edges: ", G.edges(data=True)) # Show all edges

        # print("Neighbours: ", list(G.neighbors('T-0'))) # Check neighbours of a node
        
        # print(G.nodes['laptop']) # To check node attribute
        # print(G.edges['T-0', 'laptop']) # To check edge attributes
        # print(G.number_of_nodes(), list(G.nodes)) # To check total number of nodes and list
        
        
        # print(G['person']) # Atlas view
        
        return G

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import os
# import re
# import utils
# from utils import paths_catalog as path
# import nltk
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn import model_selection
# from collections import Counter
# import spacy
# import string

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torch.autograd import Variable
# from sklearn.metrics import mean_squared_error

# from node2vec import Node2Vec
# from node2vec.edges import HadamardEmbedder

# from utils.mac import MACNetwork
# from utils.gat import GAT

# class Oracle():
#     def __init__(self, ag_video) -> None:
#         self.videos = ag_video
#         self.instances = self.videos.keys()
#         self.batch_size = 16
#         self.embed_len = 256

#         nlp_text, graph_list, answers = self.preprocess()
#         # X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.2)
#         train_ds = AGQA(nlp_text, graph_list, answers, self.embed_len)
#         self.train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
#         # val_dl = DataLoader(valid_ds, batch_size=self.batch_size)
        

#     def preprocess(self, embed=None):        
#         if not os.path.exists(path.train_path):
#             graph_features = {k: {'features': None, 'adj': None} for k in self.instances}
#             video = []; all_Q = []; all_A = []
#             for idx in self.instances:
#                 G = self.videos[idx]['G_gt']
#                 Q = self.videos[idx]['Questions']['repeat']
#                 A = self.videos[idx]['Answers']['repeat']
                
#                 all_Q.extend(Q); all_A.extend(A)
#                 video.extend([idx for i in range(len(Q))])

#                 g_embed = self.graph_embedding(idx, G, 'node2vec')
#                 graph_features[idx]['features'] = g_embed
#                 graph_features[idx]['adj'] = nx.linalg.graphmatrix.adjacency_matrix(G).todense()
#                 # all_G.extend([g_embed for i in range(len(Q))])
                
#             # assert len(all_G) == len(all_Q)
#             self.df = pd.DataFrame(list(zip(video, all_Q, all_A)), columns=['id', 'question', 'answer'])
#             utils.save_pkl(self.df, path.train_path)
#             utils.save_pkl(graph_features, path.gfeat_path)
#         else:
#             self.df = utils.load_pkl(path.train_path)
#             self.gfeat = utils.load_pkl(path.gfeat_path)
        
#         self.text_embedding()
#         self.labels = len(np.unique(self.df['answer']))
#         print("Unique answers: ", self.labels)

#         # X_nlp = list(zip(self.df['encoded'], self.df['q_length']))
#         X_nlp = list(self.df['encoded'])
#         # X_g = [self.gfeat[idx]['adj'] for idx in self.df['id']]
#         X_g = [self.videos[idx]['G_gt'] for idx in self.df['id']]
#         y = list(self.df['answer'])

#         assert len(X_g) == len(X_nlp)

#         return X_nlp, X_g, y
        
#     def text_embedding(self,):
#         tok = spacy.load('en_core_web_sm')
#         def tokenize(text):
#             text = re.sub(r'[^\w\s]?', '', str(text).lower().strip())
#             return [token.text for token in tok.tokenizer(text)]
        
#         def encode_sentence(text, vocab2index, N=10):
#             tokenized = tokenize(text)
#             encoded = np.zeros(N, dtype=int)
#             enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
#             length = min(N, len(enc1))
#             encoded[:length] = enc1[:length] # TODO Length not needed?
#             return encoded, length

#         ## TEEXT EMB #2
#         counts = Counter()
#         for index, row in self.df.iterrows():
#             counts.update(tokenize(row['question']))

#         # Delete repeating words
#         for word in list(counts):
#             if counts[word] > 5000:
#                 del counts[word]

#         #creating vocabulary
#         vocab2index = {"":0, "UNK":1}
#         self.words = ["", "UNK"]
#         for word in counts:
#             vocab2index[word] = len(self.words)
#             self.words.append(word)
#         self.df['encoded'] = self.df['question'].apply(lambda x: np.array(encode_sentence(x,vocab2index)))

#     def graph_embedding(self, idx, G, mode=None, save=False):
#         def node2vec(G):
#             show_tsne=False
#             node2vec = Node2Vec(G, dimensions=128, walk_length=5, num_walks=100, workers=4)
#             model = node2vec.fit(window=10, min_count=1, batch_words=4) 
#             node_ids = model.wv.index_to_key  # list of node IDs
#             node_embeddings = (
#                 model.wv.vectors
#             )  # numpy.ndarray of size number of nodes times embeddings dimensionality
#             # if save:
#             #     gmodel_path = os.path.join(path.g_embed_path, idx.split('.')[0] + '.emb')
#             #     if not os.path.exists(gmodel_path):
#             #         os.makedirs(gmodel_path)
#             #     model.save(gmodel_path)

#             if show_tsne:
#                 from sklearn.manifold import TSNE
#                 tsne = TSNE(n_components=2)
#                 node_embeddings_2d = tsne.fit_transform(node_embeddings)
#                 # draw the points
#                 alpha = 0.7
#                 # label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
#                 # node_colours = [label_map[target] for target in node_targets]

#                 plt.figure(figsize=(10, 8))
#                 plt.scatter(
#                     node_embeddings_2d[:, 0],
#                     node_embeddings_2d[:, 1],
#                     cmap="jet",
#                     alpha=alpha,
#                 )
#                 plt.show()
#             return node_embeddings
#             # return node_embeddings

#         if mode == 'adj':
#             return nx.adjacency_matrix(G)
#         elif mode == 'node2vec':
#             return node2vec(G)     
#         else:
#             return None           

#     def run(self, model, epochs=10, lr=0.01):
#         # lstm = LSTM(input_size=32, hidden_layer_size=64, output_size=64)
#         parameters = filter(lambda p: p.requires_grad, model.parameters())
#         optimizer = torch.optim.Adam(parameters, lr=lr)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
#         loss_func = nn.MSELoss()
        
#         for i in range(epochs):
#             model.train().to('cuda')
#             sum_loss = 0.0
#             total = 0
#             for X_nlp, X_graph, y, l in self.train_dl:
#                 X_nlp = X_nlp.long().to('cuda')
#                 X_graph = X_graph[0].float().to('cuda')
#                 y = y.float().to('cuda')
#                 y_pred = model(X_graph, X_nlp, l)
#                 optimizer.zero_grad()

#                 # loss = loss_func(y_pred.squeeze(), y)            # 1
#                 # nll = -F.log_softmax(y_pred, dim=0)                # 2 
#                 # loss = (nll * y / 10).sum(dim=0).mean()
#                 loss = F.nll_loss()                                 # 3

#                 loss.backward()
#                 optimizer.step()
#                 sum_loss += loss.item()*y.shape[0]
#                 total += y.shape[0]
            
#             scheduler.step()
#             # val_loss, val_acc, val_rmse = self.validation_metrics(model, val_dl)
#             if i % 5 == 0:
#                 # print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))
#                 print("Train: epoch %i loss %.3f, lr %.5f val loss %.3f" % (i, sum_loss/total, optimizer.param_groups[0]['lr'], 0))
        
#     def validation_metrics(self, model, valid_dl):
#         model.eval()
#         correct = 0
#         total = 0
#         sum_loss = 0.0
#         sum_rmse = 0.0
#         for x, y, l in valid_dl:
#             x = x.long()
#             y = y.long()
#             y_hat = model(x, l)
#             loss = F.cross_entropy(y_hat, y)
#             pred = torch.max(y_hat, 1)[1]
#             correct += (pred == y).float().sum()
#             total += y.shape[0]
#             sum_loss += loss.item()*y.shape[0]
#             sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
#         return sum_loss/total, correct/total, sum_rmse/total

#     def train(self,):

#         vocab_size = len(self.words)
#         # nlp_model =  LSTM(vocab_size, 24, 64)
#         # graph_model = LSTM(0, 128, 64)
#         # mac =  MACNetwork(vocab_size, 128)
#         model = GAT(nfeat=features.shape[1], 
#                 nhid=8, 
#                 nclass=self.labels + 1, 
#                 dropout=0.5, 
#                 nheads=8, 
#                 alpha=0.2)

#         self.run(mac, epochs=100, lr=0.01)
    

# class LSTM(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, ):
#         super().__init__()
#         self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
#         self.hidden = nn.Linear(16384, 512)
#         self.linear = nn.Linear(576, 1)
#         self.dropout = nn.Dropout(0.2)
#         self.act = nn.Sigmoid()
        
#     def forward(self, x, g):
#         x = self.embeddings(x)
#         # x = self.dropout(x)
#         lstm_out, (ht, ct) = self.lstm(x)
#         g = g.view(g.size(0), -1)
#         g = self.hidden(g)
#         g = self.dropout(g)
#         out = torch.cat((g,ht[-1]), dim=1)
#         out = self.linear(out)
#         return out
    
    # def create_inout_sequences(self, input_data, tw):
    #     inout_seq = []
    #     L = len(input_data)
    #     for i in range(L-tw):
    #         train_seq = input_data[i:i+tw]
    #         train_label = input_data[i+tw:i+tw+1]
    #         inout_seq.append((train_seq ,train_label))
    #     return inout_seq



        
    # def text_embedding(self,):
    #     def utils_preprocess_text(text, lst_stopwords=None):
    #         ## clean (convert to lowercase and remove punctuations and characters and then strip)
    #         text = re.sub(r'[^\w\s]?', '', str(text).lower().strip())
            
    #         lst_text = text.split()
    #         ## remove Stopwords
    #         if lst_stopwords is not None:
    #             lst_text = [word for word in lst_text if word not in 
    #                         lst_stopwords]

    #         ## Tokenize (convert from string to list)
    #         lst_text = text.split()                    
    #         ## back to string from list
    #         text = " ".join(lst_text)
    #         return text
        
    #     lst_stopwords = nltk.corpus.stopwords.words("english")
    #     self.df["text_clean"] = self.df["question"].apply(lambda x: 
    #             utils_preprocess_text(x, lst_stopwords=lst_stopwords))
    #     vectorizer = TfidfVectorizer(max_features=16, analyzer='word')
    #     corpus = self.df["text_clean"]
    #     vectorizer.fit(corpus)
    #     vectorizer.transform(corpus)
    #     dic_vocabulary = vectorizer.vocabulary_
    #     self.words = dic_vocabulary.keys()
    #     self.df['encoded'] = self.df['text_clean'].apply(lambda x: vectorizer.transform([x]).toarray())
    #     return vectorizer

    # def train(self):        
    #         from numpy import mean
    #         from numpy import std
    #         from sklearn.datasets import make_regression
    #         from sklearn.model_selection import RepeatedKFold
    #         from sklearn.linear_model import LogisticRegression
    #         # from keras import backend as K
    #         # from keras.models import Model
    #         # from keras.layers import Dense, Input, Embedding, Bidirectional, LSTM, concatenate

    #         # get the dataset
    #         def get_dataset():               
    #             # return dtf_train, y_train, dtf_test, y_test
    #             return self.df[['graph', 'text_clean']], self.df['answer'] 

    #         # # get the model
    #         # def get_model(seq_dim, g_dim):            
    #         #     nlp_input = Input(shape=(seq_dim,)) 
    #         #     g_input = Input(shape=(g_dim,))
    #         #     print(nlp_input.shape, g_input)
    #         #     # emb = Embedding(output_dim=64, input_dim=100, input_length=seq_length)(nlp_input) 
    #         #     g_out = Bidirectional(LSTM(128))(g_input) 
    #         #     g_out = Dense(24, activation='relu')(g_out) 
    #         #     concat = concatenate([nlp_input, g_out]) 
    #         #     classifier = Dense(32, activation='relu')(concat) 
    #         #     output = Dense(1, activation='sigmoid')(classifier) 
    #         #     model = Model(inputs=[nlp_input, g_input], outputs=[output])
    #         #     return model

    #         # evaluate a model using repeated k-fold cross-validation
    #         def evaluate_model(X, y):
    #             results = list()
    #             # define evaluation procedure
    #             cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    #             model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

    #             for train_ix, test_ix in cv.split(X):
    #                 # # prepare data
    #                 X_train, X_test = X.loc[train_ix], X.loc[test_ix]
    #                 y_train, y_test = y[train_ix], y.loc[test_ix]

    #                 # nlp_input = self.vectorizer.transform(X_train['text_clean'])
    #                 X_train, X_test = X_train['graph'], X_test['graph']
    #                 # print(X_train.shape, nlp_input.shape)
    #                 model.fit(X_train.to_numpy(), y_train)

    #                 # break
    #                 # define model
    #                 # model = get_model(seq_dim=nlp_input.shape[1], g_dim=64)
    #                 # # fit model
    #                 # model.fit(X_train, y_train, verbose=0, epochs=100)
    #                 # # evaluate model on test set
    #                 mae = model.evaluate(X_test, y_test, verbose=0)
    #                 # store result
    #                 print(f'MAE: {mae}')
    #                 results.append(mae)

    #             return results

    #         # load dataset
    #         X, y = get_dataset()
    #         # evaluate model
    #         results = evaluate_model(X, y)
    #         # # summarize performance
    #         # print('MAE: %.3f (%.3f)' % (mean(results), std(results)))