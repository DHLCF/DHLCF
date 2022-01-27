import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as functional
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np

import os
import argparse
import time
from tqdm import tqdm

from model import PROTOTYPE
import util as tr
import evaluate as ev


def my_collate_train(batch):
    user_id = [item[0] for item in batch]
    pos_item = [item[1] for item in batch]
    neg_item = [item[2] for item in batch]

    user_id = torch.LongTensor(user_id)
    pos_item = torch.LongTensor(pos_item)
    neg_item = torch.LongTensor(neg_item)

    return [user_id, pos_item, neg_item]

def my_collate_test(batch):
    user_id = [item[0] for item in batch]
    pos_item = [item[1] for item in batch]

    user_id = torch.LongTensor(user_id)
    pos_item = torch.LongTensor(pos_item)

    return [user_id, pos_item]

note = ''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_directory", type=str, default='yelp_process')
    parser.add_argument("--loadFilename", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default='DHLCF/data/save')

    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--epoch", type=int, default=50)

    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--n_cluster", type=int, default=100)
    parser.add_argument("--cut_loss_weight", type=float, default=0.1)


    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lambda2", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--lr_decay_every_step", type=int, default=5)
    opt = parser.parse_args()
    print(opt)

    print("Loading data >>>>>")
    user_num, item_num, interact_matrix, social_matrix, train_tri, interact_test = tr.load_all(opt.dataset_directory)
    print(user_num)
    print(item_num)
    train_dataset = tr.Train_dataset(train_tri)
    test_dataset = tr.Test_dataset(interact_test)


    print('Building dataloader >>>>>>>>>>>>>>>>>>>')
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=opt.batch_size, collate_fn=my_collate_train)
    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=opt.batch_size, collate_fn=my_collate_test)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if opt.loadFilename:
        checkpoint = torch.load(opt.loadFilename)
        sd = checkpoint['sd']
        optimizer_sd = checkpoint['opt']

    print("building model >>>>>>>>>>>>>>>")
    model = PROTOTYPE(interact_matrix, social_matrix, user_num, item_num, opt.embedding_size, opt.L, opt.n_cluster, device)

    if opt.loadFilename:
        model.load_state_dict(sd)

    for name, param in model.named_parameters():
        print(name)

    print('Building optimizers >>>>>>>')
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.lambda2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_every_step, gamma=opt.lr_decay)


    print('Start training...')
    start_epoch = 0
    if opt.loadFilename:
        checkpoint = torch.load(opt.loadFilename)
        start_epoch = checkpoint['epoch'] + 1

    model = model.to(device)
    for epoch in range(start_epoch, opt.epoch):
        model.train()

        with tqdm(total=len(train_loader), desc="epoch"+str(epoch)) as pbar:
            for index, (user_id, pos_item, neg_item) in enumerate(train_loader):
                

                new_user_embedding, new_item_embedding, loss_cut = model()

                user_id = user_id.to(device)
                pos_item = pos_item.to(device)
                neg_item = neg_item.to(device)

                user_embedded = new_user_embedding[user_id]
                pos_item_embedded = new_item_embedding[pos_item]
                neg_item_embedded = new_item_embedding[neg_item]

                pos_score = torch.mul(user_embedded, pos_item_embedded).sum(dim=-1, keepdim=False)
                neg_score = torch.mul(user_embedded, neg_item_embedded).sum(dim=-1, keepdim=False)
        
                loss = -(pos_score - neg_score).sigmoid().log().sum() + opt.cut_loss_weight * loss_cut

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(1)

        scheduler.step()


    model.eval()
    
    torch.save({
        'sd': model.state_dict(),
    }, 'model.tar')