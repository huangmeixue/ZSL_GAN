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
latent_dim = 85
feature_dim = 2048
n_critic = 5
lambda_gp = 10
LR = 0.00001
BATCH_SIZE = 64
EPOCH = 30
sample_num = 1800
batch_normalize = False
generator_hidden_size = 4096
discriminator_hidden_size = 4096
dataset = "AWA1"
class_embed_dim = 85
Model_GAN_path = "/data3/huangmeixue/ZSL_GAN/Model_WGAN/"+dataset+"/wassertein_GAN.pth"
zsl_classifier_path = "/data3/huangmeixue/ZSL_GAN/Model_WGAN/"+dataset+"/zsl_classifier.pth"
gzsl_classifier_path = "/data3/huangmeixue/ZSL_GAN/Model_WGAN/"+dataset+"/gzsl_classifier.pth"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
manualSeed = 9182
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(manualSeed)

def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)

class Generator(nn.Module):
    def __init__(self,latent_dim,class_dim,feature_dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if batch_normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + class_dim, generator_hidden_size, normalize=False),
            nn.Linear(generator_hidden_size, feature_dim),
            nn.ReLU(),
        )

    def forward(self, noise, class_embed):
        # Concatenate label embedding and latent_noise to produce input
        gen_input = torch.cat((noise, class_embed), -1)
        img_feature = self.model(gen_input)
        img_feature = img_feature.view(img_feature.size(0), -1)
        return img_feature


class Discriminator(nn.Module):
    def __init__(self,feature_dim,class_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(feature_dim + class_dim, discriminator_hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(discriminator_hidden_size, 1),
        )

    def forward(self, img_feature, class_embed):
        # Concatenate label embedding and image feature to produce input
        dis_input = torch.cat((img_feature, class_embed), -1)
        validity = self.model(dis_input)
        return validity

def compute_gradient_penalty(discriminator, real_samples, fake_samples, class_embed):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_samples.size())
    if torch.cuda.is_available():
        alpha = alpha.cuda()
    # Get random interpolation between real and fake samples
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    if torch.cuda.is_available():
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    d_interpolates = discriminator(interpolates,Variable(class_embed))

    fake_grad = torch.ones(d_interpolates.size())
    if torch.cuda.is_available():
        fake_grad = fake_grad.cuda()
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_grad,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def next_batch(data,batch_size):
    n_sample = len(data)
    idx = torch.randperm(n_sample)[0:batch_size]
    batch_feature, batch_att = data[idx]
    return batch_feature, batch_att

def train(data):
    one = torch.FloatTensor([1])
    mone = one * -1
    for step in range(0, len(data), BATCH_SIZE):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        for p in discriminator.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        for i in range(n_critic):
            feature, attribute = next_batch(data,BATCH_SIZE)
            # Sample noise as generator input
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (BATCH_SIZE, latent_dim))))
            if torch.cuda.is_available():
                real_img_feature = Variable(feature).cuda()
                attribute = Variable(attribute).cuda()
                z, one, mone = z.cuda(), one.cuda(), mone.cuda()

            optimizer_D.zero_grad()
            # Real images
            real_validity = discriminator(real_img_feature,attribute)
            real_validity = real_validity.mean()
            real_validity.backward(mone.mean())

            # Generate a batch of images
            gen_img_feature = generator(z,attribute)
            # Fake images
            dis_fake_validity = discriminator(gen_img_feature.detach(),attribute)
            dis_fake_validity = dis_fake_validity.mean()
            dis_fake_validity.backward(one.mean())
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_img_feature.data, gen_img_feature.data, attribute)
            gradient_penalty = lambda_gp * gradient_penalty
            gradient_penalty.backward()
            # Adversarial loss
            d_loss = -real_validity + dis_fake_validity + gradient_penalty
            Wasserstein_D = real_validity - dis_fake_validity
            optimizer_D.step()
        # -----------------
        #  Train Generator
        # -----------------
        for p in discriminator.parameters(): # reset requires_grad
            p.requires_grad = False # avoid computation
        
        optimizer_G.zero_grad()
        # Generate a batch of images
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (BATCH_SIZE, latent_dim))))
        if torch.cuda.is_available():
            z = z.cuda()
        gen_img_feature = generator(z,attribute)
        # Loss measures generator's ability to fool the discriminator
        # Train on fake images
        gen_fake_validity = discriminator(gen_img_feature,attribute)
        gen_fake_validity = gen_fake_validity.mean()
        g_loss = -gen_fake_validity

        g_loss.backward()
        optimizer_G.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [W loss: %f] [Grad loss: %f] [G loss: %f]"
            % (epoch+1, EPOCH, int(step/BATCH_SIZE), int(len(data)/BATCH_SIZE), d_loss.item(), Wasserstein_D.item(), gradient_penalty.item(), g_loss.item())
        )
    return g_loss,d_loss

def get_unique_vector(attributes, labels):
    # get unique class vector
    b = np.ascontiguousarray(labels).view(
        np.dtype((np.void, labels.dtype.itemsize * labels.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return np.flip(attributes[idx], 0), np.flip(labels[idx], 0)

#根据class_att生成图像特征，label与class_label相对应（class_label是one-hot表示的）
def generate_img_feature(generator,class_att,class_label):
    gen_features = np.empty((0,feature_dim),np.float32)
    gen_labels = np.empty((0,class_label.shape[1]),np.float32)
    class_num = class_label.shape[0]
    for i in range(class_num):
        # Sample noise
        z = torch.FloatTensor(np.random.normal(0, 1, (sample_num, latent_dim)))
        # Get labels ranging from 0 to class_num
        class_embed = torch.FloatTensor(np.repeat(np.expand_dims(class_att[i],axis=0),sample_num,axis=0))

        if torch.cuda.is_available():
            z = z.cuda()
            class_embed = class_embed.cuda()

        gen_img_feat = generator(z, class_embed)
        gen_features = np.vstack((gen_features,gen_img_feat.data.cpu().numpy()))
        gen_img_label = np.repeat(np.expand_dims(class_label[i],axis=0),sample_num,axis=0)
        gen_labels = np.concatenate((gen_labels,gen_img_label))
    return gen_features,gen_labels

#为了分类，将one-hot表示的标签向量转化为索引表示
def map_label(labels):
    index_labels = np.where(labels.astype(int)==1)[1]
    unique_class = np.unique(index_labels)
    class_num = len(unique_class)
    #print(unique_class,len(unique_class))
    mapped_label =  np.zeros((labels.shape[0],))
    for i in range(class_num):
        mapped_label[index_labels==unique_class[i]] = i
    mapped_label = mapped_label.astype(int)
    print("Number of Classes: {}".format(class_num))
    #print(mapped_label,class_num)
    return mapped_label

def save_model(epoch,g_net,d_net,g_loss,d_loss,model_path):
    print('===> Saving Wasserstein GAN Model...')
    state = {
        'G_state_dict': g_net.state_dict(),
        'D_state_dict': d_net.state_dict(),
        'epoch': epoch + 1,
        'g_loss': g_loss,
        'd_loss': d_loss,
    }
    #if not os.path.exists(model_path):
    #    os.makedirs(model_path)
    torch.save(state,model_path)

if __name__ == '__main__':

    ### data reading
    data = util.DATA_LOADER(dataset,preprocess,validation,generalized)

    if not os.path.exists(Model_GAN_path):
############################# Train GAN model ###############################
        features_train = torch.FloatTensor(data.features_train)
        attributes_train = torch.FloatTensor(data.attributes_train)
        train_data = Data.TensorDataset(features_train,attributes_train)
        train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)

        # Initialize generator and discriminator
        generator = Generator(latent_dim,class_embed_dim,feature_dim)
        discriminator = Discriminator(feature_dim,class_embed_dim)
        #apply函数会递归地搜索网络内的所有module并把参数表示的函数应用到所有的module上
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        print(generator)
        print(discriminator)

        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5,0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5,0.999))

        if torch.cuda.is_available():
            generator = generator.cuda()
            discriminator = discriminator.cuda()

        for epoch in range(EPOCH):
            g_loss,d_loss = train(train_data)
            save_model(epoch,generator,discriminator,g_loss,d_loss,Model_GAN_path)
    else:
########## generate image features and classification evalution #########
        generator = Generator(latent_dim,class_embed_dim,feature_dim)
        if torch.cuda.is_available():
            generator = generator.cuda()
        checkpoint = torch.load(Model_GAN_path)
        print("===> Loading Wasserstein_GAN Model... Start Epoch:{}".format(checkpoint['epoch']))
        generator.load_state_dict(checkpoint['G_state_dict'])
        
        if validation:
            # for fake data generation
            unique_attributes_val, unique_labels_val = get_unique_vector(data.attributes_val,data.labels_val)
            gen_features,gen_labels = generate_img_feature(generator,unique_attributes_val,unique_labels_val)

            gen_labels = map_label(gen_labels.astype(int))
            cls = classifier.Classifier(gen_features,gen_labels,data,zsl_classifier_path,lr=0.0001,batch_size=64,epoch=100,validation=True,generalized=False)
            unseen_acc = cls.unseen_acc
        else:
            if generalized:
                #unique_attributes_trainval, unique_labels_trainval = get_unique_vector(data.attributes_train, data.labels_train)
                #gen_features_trainval,gen_labels_trainval = generate_img_feature(generator,unique_attributes_trainval,unique_labels_trainval)
                unique_attributes_test_unseen, unique_labels_test_unseen = get_unique_vector(data.attributes_test_unseen, data.labels_test_unseen)
                gen_features_test_unseen,gen_labels_test_unseen = generate_img_feature(generator,unique_attributes_test_unseen,unique_labels_test_unseen)

                features_train = np.concatenate((data.features_train, gen_features_test_unseen), axis=0)
                labels_train = np.concatenate((data.labels_train,gen_labels_test_unseen),axis=0)
                labels_train = np.where(labels_train.astype(int)==1)[1]
                cls = classifier.Classifier(features_train,labels_train,data,gzsl_classifier_path,lr=0.0001,batch_size=64,epoch=100,validation=False,generalized=True)
                best_H, seen_acc, unseen_acc = cls.best_H, cls.seen_acc, cls.unseen_acc
            else:
                unique_attributes_test_unseen, unique_labels_test_unseen = get_unique_vector(data.attributes_test_unseen, data.labels_test_unseen)
                gen_features,gen_labels = generate_img_feature(generator,unique_attributes_test_unseen,unique_labels_test_unseen)
                gen_labels = map_label(gen_labels.astype(int))
                cls = classifier.Classifier(gen_features,gen_labels,data,zsl_classifier_path,lr=0.0001,batch_size=64,epoch=100,validation=False,generalized=False)
                unseen_acc = cls.unseen_acc