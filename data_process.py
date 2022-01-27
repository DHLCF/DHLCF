import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from copy import deepcopy

core = cpu_count()

def parallelize(data, func, num_of_processes=core):
    data_split = np.array_split(data, num_of_processes)
    with Pool(num_of_processes) as pool:
        data_list = pool.map(func, data_split)
    data = pd.concat(data_list)
    return data


def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)


def parallelize_on_rows(data, func, num_of_processes=core):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)

def get_item_count(item_list, row):
    item_count = item_list.count(row.itemid)
    return item_count

def get_all_item_counts(data, item_list):
    item_counts = parallelize_on_rows(data, partial(get_item_count, deepcopy(item_list)))
    return item_counts

def get_user_count(user_list, row):
    user_count = user_list.count(row.userid)
    return user_count

def get_all_user_counts(data, user_list):
    user_counts = parallelize_on_rows(data, partial(get_user_count, deepcopy(user_list)))
    return user_counts

def matrix_construct(dataset_name, item_fre_threshold, user_fre_threshold):
    save_dir = dataset_name + '_process'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if dataset_name == "lastfm":
        interact = pd.read_csv('lastfm/user_artists.dat', delimiter='\t', header= 0)
        interact = interact[['userID', 'artistID']]
        interact.rename(columns={'userID':'userid'}, inplace=True)
        interact.rename(columns={'artistID':'itemid'}, inplace=True)

        item_list = interact['itemid'].tolist()
        item_counts = get_all_item_counts(interact, item_list)
        interact = interact[item_counts > item_fre_threshold]

        user_list = interact['userid'].tolist()
        user_counts = get_all_user_counts(interact, user_list)
        interact = interact[user_counts > user_fre_threshold]

        social = pd.read_csv('lastfm/user_friends.dat', delimiter='\t', header= 0)
        social.rename(columns={'userID':'src'}, inplace=True)
        social.rename(columns={'friendID':'dst'}, inplace=True)

    elif dataset_name == "gowalla":
        userid = []
        itemid = []
        with open("gowalla/train.txt") as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uid = int(l[0])
                    for i in l[1:]:
                        userid.append(uid)
                        itemid.append(int(i))
        interact = pd.DataFrame({"userid":userid, "itemid":itemid})
        
        item_list = interact['itemid'].tolist()
        item_counts = get_all_item_counts(interact, item_list)
        print(item_counts.mean()/2)
        interact = interact[item_counts > item_counts.mean()/2]

        user_list = interact['userid'].tolist()
        user_counts = get_all_user_counts(interact, user_list)
        print(user_counts.mean()/2)
        interact = interact[user_counts > user_counts.mean()/2]

        social = None

    elif dataset_name == "yelp":
        userid = []
        itemid = []
        with open("yelp2018/train.txt") as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uid = int(l[0])
                    for i in l[1:]:
                        userid.append(uid)
                        itemid.append(int(i))
        interact = pd.DataFrame({"userid":userid, "itemid":itemid})
        
        item_list = interact['itemid'].tolist()
        item_counts = get_all_item_counts(interact, item_list)
        print(item_counts.mean()/2)
        interact = interact[item_counts > item_counts.mean()/2]

        user_list = interact['userid'].tolist()
        user_counts = get_all_user_counts(interact, user_list)
        print(user_counts.mean()/2)
        interact = interact[user_counts > user_counts.mean()/2]

        social = None


    user_encoder = LabelEncoder()
    if social != None:
        user_encoder.fit(pd.concat([interact['userid'],social['src'],social['dst']]))
        interact['userid'] = user_encoder.transform(interact['userid'])
        social['src'] = user_encoder.transform(social['src'])
        social['dst'] = user_encoder.transform(social['dst'])
        social.to_pickle(save_dir + "/social.pkl")
    else:
        user_encoder.fit(interact['userid'])
        interact['userid'] = user_encoder.transform(interact['userid'])

    item_encoder = LabelEncoder()
    interact['itemid'] = item_encoder.fit_transform(interact['itemid'])

    user_encoder_map = pd.DataFrame(
        {'encoded': range(len(user_encoder.classes_)), 'user': user_encoder.classes_})
    user_encoder_map.to_csv(save_dir + '/user_encoder_map.csv', index=False)

    item_encoder_map = pd.DataFrame(
        {'encoded': range(len(item_encoder.classes_)), 'item': item_encoder.classes_})
    item_encoder_map.to_csv(save_dir + '/item_encoder_map.csv', index=False)

    interact_train, interact_test = train_test_split(interact, train_size=0.9, random_state=5)
    interact_train.to_pickle(save_dir + "/interact_train.pkl")
    interact_test.to_pickle(save_dir + "/interact_test.pkl")

    print(len(user_encoder_map))
    print(len(item_encoder_map))
    return len(item_encoder_map)


def negative_sample(dataset_name, neg_num, item_num):

    save_dir = dataset_name + '_process'

    interact_train = pd.read_pickle(save_dir + '/interact_train.pkl')

    allPos = {}

    train_entry = []
    with tqdm(total=len(interact_train), desc="Building user_dic") as pbar:
        for row in interact_train.itertuples(index=False):
            user = row.userid
            item = row.itemid

            if user in allPos:
                allPos[user].append(item)
            else:
                allPos[user] = [item]
            
            pbar.update(1)

    for row in interact_train.itertuples(index=False):
        user = row.userid
        item = row.itemid

        posForUser = allPos[user]
        count = 0
        while count < neg_num:
            negitem = np.random.randint(0, item_num)
            if negitem in posForUser:
                continue
            else:
                train_entry.append([user, item, negitem])
                count += 1

    with open(save_dir + '/train_tri', "wb") as f:
        pickle.dump(train_entry, f)


item_num = matrix_construct('yelp', 12, 3)
negative_sample('yelp', 20, item_num)