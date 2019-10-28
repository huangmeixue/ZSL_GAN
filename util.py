"""
Download data and preprocess
"""

import numpy as np
import scipy.io
import os
import shutil
import re
from sklearn import preprocessing

def download_data():
    os.system('wget http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip')
    os.system('unzip xlsa17.zip')
    os.remove('xlsa17.zip')

# label和loc�?开始计�?
def prepare_data(dataset_name):
    print(dataset_name)
    mat_path = '/data3/huangmeixue/Dataset/xlsa17/data/' + dataset_name

    mat_feat = scipy.io.loadmat(mat_path + '/res101.mat')
    features = mat_feat['features'].T
    labels = mat_feat['labels']

    mat_att = scipy.io.loadmat(mat_path + '/att_splits.mat')
    #print(mat_att.keys())
    attributes = mat_att['att'].T
    
    dataset_path = '/data3/huangmeixue/Dataset/' + dataset_name
    os.makedirs(dataset_path)
    split_name = ['trainval', 'test_seen', 'test_unseen']
    for name in split_name:
        print(name)
        locs = mat_att[name + '_loc']
        features_temp = np.zeros((locs.shape[0], features.shape[1]))
        labels_temp = np.zeros((locs.shape[0], np.amax(labels)))
        attributes_temp = np.zeros((locs.shape[0], attributes.shape[1]))
        for i, loc in enumerate(locs):
            features_temp[i] = features[loc - 1]
            labels_temp[i, labels[loc - 1] - 1] = 1
            attributes_temp[i] = attributes[labels[loc - 1] - 1]
        print(features_temp.shape)
        print(labels_temp.shape)
        print(attributes_temp.shape)
        np.save(dataset_path + '/' + dataset_name + '_' + name + '_features', features_temp)
        np.save(dataset_path + '/' + dataset_name + '_' + name + '_labels', labels_temp)
        np.save(dataset_path + '/' + dataset_name + '_' + name + '_attributes', attributes_temp)
    print("=======")

def split_trainval_data(dataset_name,op='standard'):
    print(dataset_name)
    mat_path = '/data3/huangmeixue/Dataset/xlsa17/data/' + dataset_name

    mat_att = scipy.io.loadmat(mat_path + '/att_splits.mat')
    attributes = mat_att['att'].T
    #print(mat.keys())
    classnames = [classname[0][0] for classname in mat_att['allclasses_names']]
    name2label = dict(zip(classnames,range(1,len(classnames)+1)))
    #print(name2label)

    mat_feat = scipy.io.loadmat(mat_path + '/res101.mat')
    features = mat_feat['features'].T
    labels = mat_feat['labels']

    dataset_path = '/data3/huangmeixue/Dataset/' + dataset_name
    split_name = ['trainclasses2.txt', 'valclasses2.txt']
    for name in split_name:
        tr_val_classes = open(os.path.join(mat_path,name),'r').readlines()
        tr_val_labels = [name2label[name.strip('\n')] for name in tr_val_classes]

        tr_val_loc = np.array([[index] for index,label in enumerate(labels,1) if label[0] in tr_val_labels])
        if op == 'proposed':
            fake_trval_loc = set([index[0] for index in tr_val_loc])
            fake_trainval_loc = set([index[0] for index in mat_att['trainval_loc']])
            tr_val_loc = np.array(list(fake_trval_loc.intersection(fake_trainval_loc)))

        features_temp = np.zeros((tr_val_loc.shape[0], features.shape[1]))
        labels_temp = np.zeros((tr_val_loc.shape[0], np.amax(labels)))
        attributes_temp = np.zeros((tr_val_loc.shape[0], attributes.shape[1]))
        for i, loc in enumerate(tr_val_loc):
            features_temp[i] = features[loc - 1]
            labels_temp[i, labels[loc - 1] - 1] = 1
            attributes_temp[i] = attributes[labels[loc - 1] - 1]
        print(features_temp.shape)
        print(labels_temp.shape)
        print(attributes_temp.shape)
        np.save(dataset_path + '/' + dataset_name + '_' + re.sub('classes2.txt','',name) + '_features', features_temp)
        np.save(dataset_path + '/' + dataset_name + '_' + re.sub('classes2.txt','',name) + '_labels', labels_temp)
        np.save(dataset_path + '/' + dataset_name + '_' + re.sub('classes2.txt','',name) + '_attributes', attributes_temp)
    print("=======")


#data_set = ['APY', 'AWA1', 'AWA2', 'CUB', 'SUN']
#for name in data_set:
#    prepare_data(name)
#    split_trainval_data(name,'proposed')

class DATA_LOADER(object):
    def __init__(self,dataset,preprocess,validation,generalized):
        self.dataset = dataset
        self.preprocess = preprocess
        self.validation = validation
        self.generalized = generalized
        self.read_data()
        
    def read_data(self):
        if self.preprocess:
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

        if self.validation:
            self.features_train = min_max_scaler.fit_transform(np.load(os.path.join('/data3/huangmeixue/Dataset',self.dataset,self.dataset+'_train_features.npy')))
            self.attributes_train = np.load(os.path.join('/data3/huangmeixue/Dataset',self.dataset,self.dataset+'_train_attributes.npy'))
            self.labels_train = np.load(os.path.join('/data3/huangmeixue/Dataset',self.dataset,self.dataset+'_train_labels.npy'))
            self.features_val = min_max_scaler.fit_transform(np.load(os.path.join('/data3/huangmeixue/Dataset',self.dataset,self.dataset+'_val_features.npy')))
            self.attributes_val = np.load(os.path.join('/data3/huangmeixue/Dataset',self.dataset,self.dataset+'_val_attributes.npy'))
            self.labels_val = np.load(os.path.join('/data3/huangmeixue/Dataset',self.dataset,self.dataset+'_val_labels.npy'))
        else:
            self.features_train = min_max_scaler.fit_transform(np.load(os.path.join('/data3/huangmeixue/Dataset',self.dataset,self.dataset+'_trainval_features.npy')))
            self.attributes_train = np.load(os.path.join('/data3/huangmeixue/Dataset',self.dataset,self.dataset+'_trainval_attributes.npy'))  
            self.labels_train = np.load(os.path.join('/data3/huangmeixue/Dataset',self.dataset,self.dataset+'_trainval_labels.npy'))
            self.features_test_unseen = min_max_scaler.fit_transform(np.load(os.path.join('/data3/huangmeixue/Dataset',self.dataset,self.dataset+'_test_unseen_features.npy')))
            self.attributes_test_unseen = np.load(os.path.join('/data3/huangmeixue/Dataset',self.dataset,self.dataset+'_test_unseen_attributes.npy'))
            self.labels_test_unseen = np.load(os.path.join('/data3/huangmeixue/Dataset',self.dataset,self.dataset+'_test_unseen_labels.npy'))
            #unique_attributes_test_unseen, unique_labels_test_unseen = get_unique_vector(attributes_test_unseen, labels_test_unseen)
            if self.generalized:
                self.features_test_seen = min_max_scaler.fit_transform(np.load(os.path.join('/data3/huangmeixue/Dataset',self.dataset,self.dataset+'_test_seen_features.npy')))
                self.attributes_test_seen = np.load(os.path.join('/data3/huangmeixue/Dataset',self.dataset,self.dataset+'_test_seen_attributes.npy'))            
                self.labels_test_seen = np.load(os.path.join('/data3/huangmeixue/Dataset',self.dataset,self.dataset+'_test_seen_labels.npy'))


