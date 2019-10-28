#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F #激励函数
from torch.autograd import Variable
import numpy as np
import time
import os
import random
from sklearn import preprocessing
import classifier
import util

validation = False
generalized = True
preprocess = True
feature_dim = 2048
LR = 0.001
BATCH_SIZE = 100
EPOCH = 50
VAL_SIZE = 0.2
shuffle = True
random_seed = 9182
dataset = "AWA1"
Model_GAN_path = "/data3/huangmeixue/ZSL_GAN/Model_CLSWGAN/"+dataset+"/CLSW_GAN.pth"
PREFINE_CLASS_path = "/data3/huangmeixue/ZSL_GAN/Model_CLSWGAN/"+dataset+"/prefine_class.pth"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        torch.nn.init.normal_(m.weight.data, 0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

class SoftMax_Classifier(nn.Module):  # 继承 torch 的 Module
    def __init__(self, feature_dim, class_num):
        super(SoftMax_Classifier, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.out = torch.nn.Linear(feature_dim, class_num)  

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = self.out(x) # # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x

def map_label(labels):
	index_labels = np.where(labels==1)[1]
	unique_class = np.unique(index_labels)
	class_num = len(unique_class)
	#print(unique_class,len(unique_class))
	mapped_label =  np.zeros((labels.shape[0],))
	for i in range(class_num):
		mapped_label[index_labels==unique_class[i]] = i
	mapped_label = mapped_label.astype(int)
	print("Number of Classes: {}".format(class_num))
	return mapped_label

def accuracy(true_label, predicted_label):
	unique_class = np.unique(true_label)
	class_num = len(unique_class)
	acc_per_class = np.zeros(class_num)
	for i in range(class_num):
		class_index = [index for index,label in enumerate(true_label) if label==unique_class[i]]
		acc_per_class[i] = float(np.sum([1 for idx in class_index if true_label[idx]==predicted_label[idx]])) / float(len(class_index))
	return np.mean(acc_per_class)

def save_model(model,epoch,train_loss,train_acc,val_loss,val_acc,best_acc):
	print('===> Saving Model...')
	state = {
		'state_dict': model.state_dict(),
		'epoch': epoch + 1,
		'train_loss': train_loss,
		'train_acc': train_acc,
		'val_loss': val_loss,
		'val_acc': val_acc,
		'best_acc': best_acc,
	}

	torch.save(state,PREFINE_CLASS_path)

def load_model(model):
	print('===> Loading Model...')
	if os.path.exists(PREFINE_CLASS_path):
		checkpoint = torch.load(PREFINE_CLASS_path)
		model.load_state_dict(checkpoint['state_dict'])        # 从字典中依次读取
		start_epoch = checkpoint['epoch']
		best_acc = checkpoint['best_acc']
		print('===> Load last checkpoint data')
	else:
		start_epoch = 0
		best_acc = 0
		print('===> Start from scratch')
	print("Start Epoch {}, Best Accuracy:{:.4f}".format(start_epoch,best_acc))
	return model,start_epoch,best_acc

#划分训练集验证集，并且加载dataloader
def load_data(features,labels):
	# 先转换成 torch 能识别的 Dataset
	features = torch.FloatTensor(features)
	labels = torch.LongTensor(labels)
	dataset = Data.TensorDataset(features,labels)
	#划分训练集验证集
	# Creating data indices for training and validation splits:
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	split = int(np.floor(VAL_SIZE * dataset_size))
	if shuffle:
		np.random.seed(random_seed)
		torch.manual_seed(random_seed)
		np.random.shuffle(indices)
	train_indices, val_indices = indices[split:], indices[:split]
	# Creating data samplers and loaders:
	train_sampler = Data.sampler.SubsetRandomSampler(train_indices)
	val_sampler = Data.sampler.SubsetRandomSampler(val_indices)
	# 把 dataset 放入 DataLoader
	train_loader = Data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE,sampler=train_sampler)
	val_loader = Data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE,sampler=val_sampler)
	
	return train_loader,val_loader

def train(trainloader, model, criterion, optimizer, epoch):
	model.train()
	time_start = time.time()
	train_loss = []
	pred_label = []
	true_label = []
	for step, (batch_x, batch_y) in enumerate(trainloader):  # 每一步 loader 释放一小批数据用来学习
		batch_x = Variable(batch_x)
		batch_y = Variable(batch_y)
		if torch.cuda.is_available():
			batch_x = batch_x.cuda()
			batch_y = batch_y.cuda()
		out_y = model(batch_x) # 喂给 model 训练数据 x, 输出分析值
		batch_loss = criterion(out_y,batch_y)# 计算两者的误差
		optimizer.zero_grad()   # 清空上一步的残余更新参数值
		batch_loss.backward()         # 误差反向传播, 计算参数更新值
		optimizer.step()        # 将参数更新值施加到 mdoel 的 parameters 上
		train_loss.append(batch_loss.data.cpu().numpy().squeeze())
		pred_label.extend(torch.max(F.softmax(out_y,dim=1),1)[1].data.cpu().numpy().squeeze())
		true_label.extend(batch_y.data.cpu().numpy().squeeze())
	train_loss = np.mean(train_loss)
	train_acc = accuracy(true_label,pred_label)
	epoch_time = time.time() - time_start
	print("Epoch {}, Train Loss:{:.4f}, Train Accuracy:{:.4f}, Time: {:.4f}".format(epoch+1,train_loss,train_acc,epoch_time))
	#log.write("Epoch "+str(epoch+1)+", Train Loss: "+str(round(train_loss,4))+", Train Accuracy: "+str(round(train_acc,4))+", Time: "+str(round(epoch_time,4))+"\n")
	return train_loss,train_acc

def validate(valloader, model, criterion, optimizer, epoch):
	model.eval()
	time_start = time.time()
	val_loss = []
	pred_label = []
	true_label = []
	for step, (batch_x, batch_y) in enumerate(valloader):  # 每一步 loader 释放一小批数据用来学习
		batch_x = Variable(batch_x)
		batch_y = Variable(batch_y)
		if torch.cuda.is_available():
			batch_x = batch_x.cuda()
			batch_y = batch_y.cuda()
		out_y = model(batch_x) # 喂给 model 训练数据 x, 输出分析值
		batch_loss = criterion(out_y,batch_y)# 计算两者的误差

		val_loss.append(batch_loss.data.cpu().numpy().squeeze())
		pred_label.extend(torch.max(F.softmax(out_y,dim=1),1)[1].data.cpu().numpy().squeeze())
		true_label.extend(batch_y.data.cpu().numpy().squeeze())
	val_loss = np.mean(val_loss)
	val_acc = accuracy(true_label,pred_label)
	epoch_time = time.time() - time_start
	print("Epoch {}, Val Loss:{:.4f}, Val Accuracy:{:.4f}, Time: {:.4f}".format(epoch+1,val_loss,val_acc,epoch_time))
	#log.write("Epoch "+str(epoch+1)+", Val Loss: "+str(round(val_loss,4))+", Val Accuracy: "+str(round(val_acc,4))+", Time: "+str(round(epoch_time,4))+"\n")
	return val_loss,val_acc

def test(model,features,labels):
	
	features = torch.FloatTensor(features)
	labels = torch.LongTensor(labels)
	dataset = Data.TensorDataset(features,labels)
	test_loader = Data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE)
	
	model,start_epoch,best_acc = load_model(model)

	pred_label = []
	true_label = []
	for step, (batch_x, batch_y) in enumerate(test_loader):
		batch_x = Variable(batch_x)
		batch_y = Variable(batch_y)
		if torch.cuda.is_available():
			batch_x = batch_x.cuda()
			batch_y = batch_y.cuda()
		out_y = model(batch_x) # 喂给 model 训练数据 x, 输出分析值

		pred_label.extend(torch.max(F.softmax(out_y,dim=1),1)[1].data.cpu().numpy().squeeze())
		true_label.extend(batch_y.data.cpu().numpy().squeeze())

	test_acc = accuracy(true_label,pred_label)
	print("Test Accuracy:{:.4f}".format(test_acc))
	return test_acc	

if __name__ == '__main__':

	### data reading
	data = util.DATA_LOADER(dataset,preprocess,validation,generalized)
	if not os.path.exists(PREFINE_CLASS_path):
		labels_train = map_label(data.labels_train)
		class_num = len(np.unique(labels_train))

		train_loader,val_loader = load_data(data.features_train,labels_train)

		model = SoftMax_Classifier(feature_dim=feature_dim,class_num=class_num)
		model.apply(weights_init)
		print(model)
		# optimizer 是训练的工具
		optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5,0.999))  # 传入 model 的所有参数, 学习率
		criterion = torch.nn.CrossEntropyLoss()

		if torch.cuda.is_available():
			model = model.cuda()
			criterion = criterion.cuda()

		best_acc = 0
		for epoch in range(EPOCH):
			# train for one epoch
			train_loss,train_acc = train(train_loader,model,criterion,optimizer,epoch)
			# evaluate on validation set
			val_loss,val_acc = validate(val_loader,model,criterion,optimizer,epoch)
			if val_acc > best_acc:
				best_acc = max(val_acc,best_acc)
				save_model(model,epoch,train_loss,train_acc,val_loss,val_acc,best_acc)
			print("Best Accuracy:{:.4f}".format(best_acc))
	else:
		if not validation:
			labels_test_seen = map_label(data.labels_test_seen)
			class_num = len(np.unique(labels_test_seen))
			model = SoftMax_Classifier(feature_dim=feature_dim,class_num=class_num)
			if torch.cuda.is_available():
				model = model.cuda()
			test_acc = test(model,data.features_test_seen,labels_test_seen)