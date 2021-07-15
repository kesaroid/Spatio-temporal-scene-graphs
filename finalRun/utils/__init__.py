# from utils.dataset import AGDataset
# from utils.qa import QA
# from utils.qa import create_questions

import pickle

def save_pkl(data, name):
    with open(f'{name}', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(name):
    with open(f'{name}', 'rb') as handle:
        data = pickle.load(handle)
    return data
