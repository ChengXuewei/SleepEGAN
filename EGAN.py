##加载包
import numpy as np
import re
import os
import importlib
import glob
import argparse
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from sklearn import preprocessing



img_shape=3000;
latent_dim=100;
n_filters=128


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model1 = nn.Sequential(
            
            ##Coarse grained feature extraction in 5 steps
            nn.Upsample(scale_factor=30),
            
            nn.Conv1d(1,n_filters,50,stride=6,padding=22),
            nn.BatchNorm1d(n_filters,eps=0.001,momentum=0.99),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.MaxPool1d(8,stride=8,padding=4),
            nn.Dropout(p=0.5),
            
            ##Eleven steps for fine-grained feature extraction+LSTM
            nn.Conv1d(n_filters,n_filters,8,stride=1,padding=4),
            nn.BatchNorm1d(n_filters,eps=0.001,momentum=0.99),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(n_filters,n_filters,8,stride=1,padding=3),
            nn.BatchNorm1d(n_filters,eps=0.001,momentum=0.99),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(n_filters,n_filters,8,stride=1,padding=3),
            nn.BatchNorm1d(n_filters,eps=0.001,momentum=0.99),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.MaxPool1d(4,stride=4,padding=2),
            nn.Dropout(p=0.5),
            
            nn.LSTM(16,n_filters,1)
        )
        
        self.model2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(n_filters,img_shape),
            nn.Tanh()
        )
    
    def forward(self, z):
        z=z.reshape((z.shape[0],1,latent_dim))
        img1=self.model1(z)[0]
        img2=self.model2(img1[:,-1,:])
        img=img2
        img = img.view(img.size(0), img_shape)
        
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model1 = nn.Sequential(
            nn.Conv1d(1,n_filters,50,stride=6,padding=22),
            nn.BatchNorm1d(n_filters,eps=0.001,momentum=0.99),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(8,stride=8,padding=4),
            nn.Dropout(p=0.5),
            
            nn.Conv1d(n_filters,n_filters,8,stride=1,padding=4),
            nn.BatchNorm1d(n_filters,eps=0.001,momentum=0.99),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(n_filters,n_filters,8,stride=1,padding=3),
            nn.BatchNorm1d(n_filters,eps=0.001,momentum=0.99),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(n_filters,n_filters,8,stride=1,padding=3),
            nn.BatchNorm1d(n_filters,eps=0.001,momentum=0.99),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.MaxPool1d(4,stride=4,padding=2),
            nn.Dropout(p=0.5),
            
            nn.LSTM(16,n_filters,1)
        )
        
        self.model2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(n_filters,1),
            nn.Sigmoid()
        )
        

    def forward(self, img):
        img_flat = img.view(img.size(0), 1, img_shape)
        img1= self.model1(img_flat)[0]
        img2= self.model2(img1[:,-1,:])
        validity=img2

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

cuda = True if torch.cuda.is_available() else False;cuda

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    
# Optimizers
l_rate=0.0002 ##adam: learning rate
b1=0.5    ##adam: decay of first order momentum of gradient
b2=0.999  ##adam: decay of first order momentum of gradient
optimizer_G = torch.optim.Adam(generator.parameters(), lr=l_rate, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=l_rate, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def gan_x(train_x,train_y,stage_class,type):
    # Configure data loader
    latent_dim=100;
    img_shape=3000;
    if type==1:
        batch_size=32;
    else:
        batch_size=16;
    n_epochs=len(train_x)*4
    # Configure data loader
    train_X=np.empty(shape=(0,img_shape))
    for j in range(0,len(train_x)):
        X=np.array(train_x[j]);
        X=X.reshape(X.shape[0],img_shape)
        ID=np.where(train_y[j]==stage_class);
        X=X[ID,:].reshape(X[ID,:].shape[1],img_shape);
        train_X=np.vstack((train_X,X))
    ## Perform data normalization because GAN performs better on normalized data【-1，1】
    max_abs_scaler = preprocessing.MaxAbsScaler()
    train_X = max_abs_scaler.fit_transform(train_X)
    ##Load data into the dataloader
    dataloader =torch.utils.data.DataLoader(train_X,batch_size=batch_size,drop_last=True,shuffle=True);len(dataloader)
    generate_x=[]
    for epoch in range(n_epochs):
        for i, imgs in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0],latent_dim))))

            # Generate a batch of images

            gen_imgs= generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()


            if ((i==(len(dataloader)-1)) & (epoch>=round(n_epochs*0.5))):
                jj=int(epoch % len(train_x))
                fs=sum(discriminator(gen_imgs)>0.5)  ##The number of successful counterfeits
                print(
                   "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Fake success: %d]"
                    % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), fs)
                )
                gan_x=gen_imgs.clone().detach().cpu().numpy();
                Gan_x=max_abs_scaler.inverse_transform(gan_x) ##Invert back
                Gan_x=Gan_x.reshape(len(Gan_x),img_shape,1,1);Gan_x.shape
                if len(generate_x)<len(train_x):
                    generate_x.append(Gan_x)
                else:
                    generate_x[jj]=np.vstack((generate_x[jj],Gan_x)) 
                    
            batches_done = epoch * len(dataloader) + i

    return generate_x