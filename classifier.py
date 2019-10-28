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

def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        torch.nn.init.normal_(m.weight.data, 0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
        
class SoftMax_Classifier(nn.Module):  # 继承 torch 的 Module
    def __init__(self, feature_dim, class_num):
        super(SoftMax_Classifier, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.classifier = nn.Linear(feature_dim, class_num)  

    def forward(self, feature):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        prod = self.classifier(feature) # # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return prod

class Classifier(object):
	def __init__(self,features_train,labels_train,data,classifier_path,lr=0.001,batch_size=100,epoch=20,validation=False,generalized=True):
		self.features_train = features_train
		self.labels_train = labels_train
		self.classifier_path = classifier_path
		self.lr = lr
		self.batch_size = batch_size
		self.epoch = epoch
		self.feature_dim = features_train.shape[1]
		self.allclass_num = len(np.unique(labels_train))

		if validation:
			self.features_test = data.features_val
			self.labels_test = self.map_label(data.labels_val)
		else:
			if generalized:
				self.features_test_unseen = data.features_test_unseen
				self.labels_test_unseen = np.where(data.labels_test_unseen==1)[1]
				self.features_test_seen = data.features_test_seen
				self.labels_test_seen = np.where(data.labels_test_seen==1)[1]
			else:
				self.features_test = data.features_test_unseen
				self.labels_test = self.map_label(data.labels_test_unseen)
			#self.unseenclass_num = len(np.unique(data.labels_test_unseen))
			#self.seenclass_num = len(np.unique(data.labels_test_seen))

		self.model = SoftMax_Classifier(feature_dim=self.feature_dim,class_num=self.allclass_num)
		self.model.apply(weights_init)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5,0.999))  # 传入 model 的所有参数, 学习率
		self.criterion = torch.nn.CrossEntropyLoss()
		if torch.cuda.is_available():
			self.model, self.criterion = self.model.cuda(),self.criterion.cuda()
		
		if generalized:
			self.best_H, self.seen_acc, self.unseen_acc = self.fit_gzsl()
		else:
			self.unseen_acc = self.fit_zsl()

	def map_label(self,onehot_labels):
		index_labels = np.where(onehot_labels.astype(int)==1)[1]
		unique_labels = np.unique(index_labels)
		mapped_index_labels = np.empty(index_labels.shape[0],)
		for i in range(len(unique_labels)):
			mapped_index_labels[index_labels==unique_labels[i]] = i
		mapped_index_labels = mapped_index_labels.astype(int)
		return mapped_index_labels

	def fit_gzsl(self):
		if not os.path.exists(self.classifier_path):
			features = torch.FloatTensor(self.features_train)
			labels = torch.LongTensor(self.labels_train)
			dataset = Data.TensorDataset(features,labels)
			dataloader = Data.DataLoader(dataset=dataset,batch_size=self.batch_size,shuffle=True)

			best_H = 0
			for epoch_n in range(self.epoch):
				self.model.train()
				time_start = time.time()
				train_loss = []
				pred_label = []
				true_label = []
				for step, (batch_x, batch_y) in enumerate(dataloader):  # 每一步 loader 释放一小批数据用来学习
					batch_x = Variable(batch_x)
					batch_y = Variable(batch_y)
					if torch.cuda.is_available():
						batch_x = batch_x.cuda()
						batch_y = batch_y.cuda()
					out_y = self.model(batch_x) # 喂给 model 训练数据 x, 输出分析值
					batch_loss = self.criterion(out_y,batch_y)# 计算两者的误差
					self.optimizer.zero_grad()   # 清空上一步的残余更新参数值
					batch_loss.backward()         # 误差反向传播, 计算参数更新值
					self.optimizer.step()        # 将参数更新值施加到 mdoel 的 parameters 上
					train_loss.append(batch_loss.data.cpu().numpy().squeeze())
					pred_label.extend(torch.max(F.softmax(out_y,dim=1),1)[1].data.cpu().numpy().squeeze())
					true_label.extend(batch_y.data.cpu().numpy().squeeze())
				train_loss = np.mean(train_loss)
				train_acc = self.accuracy(true_label,pred_label)
				epoch_time = time.time() - time_start
				print("Epoch {}, Train Loss:{:.4f}, Train Accuracy:{:.4f}, Time: {:.4f}".format(epoch_n+1,train_loss,train_acc,epoch_time))
				#log.write("Epoch "+str(epoch+1)+", Train Loss: "+str(round(train_loss,4))+", Train Accuracy: "+str(round(train_acc,4))+", Time: "+str(round(epoch_time,4))+"\n")
				seen_loss,seen_acc = self.validate(epoch_n,self.features_test_seen,self.labels_test_seen)
				unseen_loss,unseen_acc = self.validate(epoch_n,self.features_test_unseen,self.labels_test_unseen)
				H = 2*seen_acc*unseen_acc / (seen_acc+unseen_acc)

				if H > best_H:
					start_epoch = epoch_n + 1
					best_H = max(H,best_H)
					state = {'state_dict':self.model.state_dict(),'epoch':epoch_n+1,'best_H':best_H,'seen_acc':seen_acc,'unseen_acc':unseen_acc}
					torch.save(state,self.classifier_path)
		else:
			checkpoint = torch.load(self.classifier_path)
			self.model.load_state_dict(checkpoint['state_dict'])
			start_epoch = checkpoint['epoch']
			best_H = checkpoint['best_H']
			seen_acc = checkpoint['seen_acc']
			unseen_acc = checkpoint['unseen_acc']

			print("Start Epoch {}, Best H:{:.4f}, seen_acc:{:.4f}, unseen_acc:{:.4f}".format(start_epoch,best_H,seen_acc,unseen_acc))
		return best_H,seen_acc,unseen_acc

	def fit_zsl(self):
		if not os.path.exists(self.classifier_path):
			features = torch.FloatTensor(self.features_train)
			labels = torch.LongTensor(self.labels_train)
			dataset = Data.TensorDataset(features,labels)
			dataloader = Data.DataLoader(dataset=dataset,batch_size=self.batch_size,shuffle=True)

			best_acc = 0
			for epoch_n in range(self.epoch):
				self.model.train()
				time_start = time.time()
				train_loss = []
				pred_label = []
				true_label = []
				for step, (batch_x, batch_y) in enumerate(dataloader):  # 每一步 loader 释放一小批数据用来学习
					batch_x = Variable(batch_x)
					batch_y = Variable(batch_y)
					if torch.cuda.is_available():
						batch_x = batch_x.cuda()
						batch_y = batch_y.cuda()
					out_y = self.model(batch_x) # 喂给 model 训练数据 x, 输出分析值
					batch_loss = self.criterion(out_y,batch_y)# 计算两者的误差
					self.optimizer.zero_grad()   # 清空上一步的残余更新参数值
					batch_loss.backward()         # 误差反向传播, 计算参数更新值
					self.optimizer.step()        # 将参数更新值施加到 mdoel 的 parameters 上
					train_loss.append(batch_loss.data.cpu().numpy().squeeze())
					pred_label.extend(torch.max(F.softmax(out_y,dim=1),1)[1].data.cpu().numpy().squeeze())
					true_label.extend(batch_y.data.cpu().numpy().squeeze())
				train_loss = np.mean(train_loss)
				train_acc = self.accuracy(true_label,pred_label)
				epoch_time = time.time() - time_start
				print("Epoch {}, Train Loss:{:.4f}, Train Accuracy:{:.4f}, Time: {:.4f}".format(epoch_n+1,train_loss,train_acc,epoch_time))
				#log.write("Epoch "+str(epoch+1)+", Train Loss: "+str(round(train_loss,4))+", Train Accuracy: "+str(round(train_acc,4))+", Time: "+str(round(epoch_time,4))+"\n")
				val_loss,val_acc = self.validate(epoch_n,self.features_test,self.labels_test)
				if val_acc > best_acc:
					start_epoch = epoch_n + 1
					best_acc = max(val_acc,best_acc)
					state = {'state_dict':self.model.state_dict(),'epoch':epoch_n+1,'best_acc':best_acc}
					torch.save(state,self.classifier_path)
		else:
			checkpoint = torch.load(self.classifier_path)
			self.model.load_state_dict(checkpoint['state_dict'])
			start_epoch = checkpoint['epoch']
			best_acc = checkpoint['best_acc']
			#_,val_acc = self.validate(start_epoch-1)

			print("Start Epoch {}, Best Accuracy:{:.4f}".format(start_epoch,best_acc))
		return best_acc

	def validate(self,epoch_n,features,labels):
		features = torch.FloatTensor(features)
		labels = torch.LongTensor(labels)
		dataset = Data.TensorDataset(features,labels)
		dataloader = Data.DataLoader(dataset=dataset,batch_size=self.batch_size,shuffle=True)

		self.model.eval()
		time_start = time.time()
		val_loss = []
		pred_label = []
		true_label = []
		for step, (batch_x, batch_y) in enumerate(dataloader):  # 每一步 loader 释放一小批数据用来学习
			batch_x = Variable(batch_x)
			batch_y = Variable(batch_y)
			if torch.cuda.is_available():
				batch_x = batch_x.cuda()
				batch_y = batch_y.cuda()
			out_y = self.model(batch_x) # 喂给 model 训练数据 x, 输出分析值
			batch_loss = self.criterion(out_y,batch_y)# 计算两者的误差
			val_loss.append(batch_loss.data.cpu().numpy().squeeze())
			pred_label.extend(torch.max(F.softmax(out_y,dim=1),1)[1].data.cpu().numpy().squeeze())
			true_label.extend(batch_y.data.cpu().numpy().squeeze())
		val_loss = np.mean(val_loss)
		val_acc = self.accuracy(true_label,pred_label)
		epoch_time = time.time() - time_start
		print("Epoch {}, Val Loss:{:.4f}, Val Accuracy:{:.4f}, Time: {:.4f}".format(epoch_n+1,val_loss,val_acc,epoch_time))
		#log.write("Epoch "+str(epoch+1)+", Val Loss: "+str(round(val_loss,4))+", Val Accuracy: "+str(round(val_acc,4))+", Time: "+str(round(epoch_time,4))+"\n")
		return val_loss,val_acc

	def accuracy(self, true_label, predicted_label):
		unique_class = np.unique(true_label)
		acc_per_class = np.zeros(len(unique_class))
		for i in range(len(unique_class)):
			class_index = [index for index,label in enumerate(true_label) if label==unique_class[i]]
			acc_per_class[i] = float(np.sum([1 for idx in class_index if true_label[idx]==predicted_label[idx]])) / float(len(class_index))
		return np.mean(acc_per_class)
