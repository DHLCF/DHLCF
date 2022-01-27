import torch
from torch.utils.data.dataset import Dataset
import os
from io import open
import pickle
import numpy as np
import random
import pandas as pd

def load_all(directory):

    item_encoder_map = pd.read_csv(directory + '/item_encoder_map.csv')
    item_num = len(item_encoder_map)

    user_encoder_map = pd.read_csv(directory + '/user_encoder_map.csv')
    user_num = len(user_encoder_map)
    
    interact_train = pd.read_pickle(directory + '/interact_train.pkl')
    interact_test = pd.read_pickle(directory + '/interact_test.pkl')

    index = [interact_train['userid'].tolist(), interact_train['itemid'].tolist()]
    value = [1] * len(interact_train)
    interact_matrix = torch.sparse_coo_tensor(index, value, (user_num, item_num))

    social_matrix = None

    with open(directory + '/train_tri', "rb") as f:
        train_tri = pickle.load(f)

    return user_num, item_num, interact_matrix, social_matrix, train_tri, interact_test




class Train_dataset(Dataset):
    def __init__(self, train_tri):
        super(Train_dataset, self).__init__()
        self.train_tri = train_tri

    def __len__(self):
        return len(self.train_tri)

    def __getitem__(self, idx):
        entry = self.train_tri[idx]

        user = entry[0]
        pos_item = entry[1]
        neg_item = entry[2]

        return user, pos_item, neg_item


class Test_dataset(Dataset):
    def __init__(self, interact_test):
        super(Test_dataset, self).__init__()

        self.test_tri = interact_test

    def __len__(self):
        return len(self.test_tri)

    def __getitem__(self, idx):
        entry = self.test_tri.iloc[idx]

        user = entry.userid
        item = entry.itemid

        return user, item


