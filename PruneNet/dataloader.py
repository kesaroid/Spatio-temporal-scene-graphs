import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np

import networkx as nx

class AGQA(Dataset):
    def __init__(self, X_t, X_g, Y, embed_len):
        self.X_t = X_t
        self.X_g = X_g
        self.y = Y
        self.embed_len = embed_len
        self.padding = False
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx): # Category features
        def get_features():
            # A = nx.linalg.graphmatrix.adjacency_matrix(self.X_g[idx]).todense()
            # I = np.matrix(np.eye(A.shape[0]))
            # X = np.matrix([
            #     [1]
            #     for i in range(A.shape[0])], dtype=float)
            # # X = []
            # # for i in list(self.X_g[idx].nodes()):
            # #     if isinstance(i, str):
            # #         X.append(label_to_idx[i])
            # #     else:
            # #         X.append(i)
            # # X = np.expand_dims(np.array(X), axis=1)
            # A_hat = A + I
            # D = np.array(np.sum(A_hat, axis=0))[0]
            # D = np.matrix(np.diag(D))
            # print(A, '\n', D**-1)
            # return D**-1 * A_hat * X, A

            A = nx.linalg.graphmatrix.adjacency_matrix(self.X_g[idx]).todense()
            # I = np.matrix(np.eye(A.shape[0]))
            # A = A + I
            X = np.sum(A, axis=0)
            print(self.X_g[idx].nodes(data=True))
            print(self.X_g[idx].edges(data=True))
            print(A)
            print(X)
            return A[:, np.argmax(X)]

        X_t = torch.from_numpy(self.X_t[idx][0].astype(np.int32)).squeeze()
        # X_t = torch.from_numpy(self.X_t[idx])
        X_g= get_features()
        X_g = torch.from_numpy(X_g)
        X_adj = torch.zeros_like(X_g)

        print(X_g, self.y[idx])
        if self.padding:
            pad = torch.zeros((self.embed_len - X_g.shape[0], X_g.shape[1])).double()
            X_g = torch.cat((X_g.double(), pad), dim=0)
        return X_t, X_g, X_adj, self.y[idx] # l: self.X_t[idx][1]

    # def __getitem__(self, idx): # Using Node2vec
    #     X_t = torch.from_numpy(self.X_t[idx][0].astype(np.int32)).squeeze()
    #     X_g, adj = self.X_g[idx][0], self.X_g[idx][1]
    #     X_g = torch.from_numpy(X_g)
    #     X_adj = torch.from_numpy(adj)
    #     if self.padding:
    #         pad = torch.zeros((self.embed_len - X_g.shape[0], X_g.shape[1])).double()
    #         X_g = torch.cat((X_g.double(), pad), dim=0)
        
    #     return X_t, X_g, X_adj, self.y[idx] # l: self.X_t[idx][1]

label_to_idx = {
        "person": 1001,
        "bag": 1002,
        "bed": 1003,
        "blanket": 1004,
        "book": 1005,
        "box": 1006,
        "broom": 1007,
        "chair": 1008,
        "closet/cabinet": 1009,
        "clothes": 1010,
        "cup/glass/bottle": 1011,
        "dish": 1012,
        "door": 1013,
        "doorknob": 1014,
        "doorway": 1015,
        "floor": 1016,
        "food": 1017,
        "groceries": 1018,
        "laptop": 1019,
        "light": 1020,
        "medicine": 1021,
        "mirror": 1022,
        "paper/notebook": 1023,
        "phone/camera": 1024,
        "picture": 1025,
        "pillow": 1026,
        "refrigerator": 1027,
        "sandwich": 1028,
        "shelf": 1029,
        "shoe": 1030,
        "sofa/couch": 1031,
        "table": 1032,
        "television": 1033,
        "towel": 1034,
        "vacuum": 1035,
        "window": 1036,
        "looking_at": 1037,
        "not_looking_at": 1038,
        "unsure": 1039,
        "above": 1040,
        "beneath": 1041,
        "in_front_of": 1042,
        "behind": 1043,
        "on_the_side_of": 1044,
        "in": 1045,
        "carrying": 1046,
        "covered_by": 1047,
        "drinking_from": 1048,
        "eating": 1049,
        "have_it_on_the_back": 1050,
        "holding": 1051,
        "leaning_on": 1052,
        "lying_on": 1053,
        "not_contacting": 1054,
        "other_relationship": 1055,
        "sitting_on": 1056,
        "standing_on": 1057,
        "touching": 1058,
        "twisting": 1059,
        "wearing": 1060,
        "wiping": 1061,
        "writing_on": 1062
    }

# def network_collate(batch):
#     X_t = torch.from_numpy(np.ndarray([item[0] for item in batch]))
#     X_g = torch.from_numpy([item[1] for item in batch])
#     X_adj = torch.from_numpy([item[2] for item in batch])
    
#     target = torch.from_numpy([item[3] for item in batch])
    
#     print(X_adj[0].shape)
#     print(X_adj)
#     print(len(X_adj))

#     return [X_t, X_g, X_adj, target]