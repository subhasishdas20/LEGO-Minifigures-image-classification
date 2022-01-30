#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F

import torchvision 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from PIL import Image


# In[2]:


train_path = r"D:\Python class\LEGO"
test_path = r"D:\Python class\test"


# ## IMAGE TRANSFORM

# In[3]:


img_size = 120 
img_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# In[4]:


img_data =ImageFolder(root = train_path, transform = img_transform)


# In[5]:


img_data.classes


# In[6]:


len(img_data)


# In[7]:


train_data, val_data, test_data= random_split(img_data, [330,50,25])


# In[8]:


batch_size = 32

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True)
val_loader = DataLoader(val_data, batch_size, shuffle = False)


# In[9]:


for image, label in train_loader:
    print(image.shape, label.shape)
    break


# In[10]:


def show_image(data):
    for images, labels in data:
        plt.figure(figsize = (13,8))
        plt.imshow(make_grid(images).permute(1,2,0))
        plt.show()
        break


# In[11]:


show_image(train_loader)


# In[12]:


show_image(val_loader)


# # CNN

# In[13]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn. Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(28*28*64,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,151)
        
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.conv1(x) #118*118*32
        x = self.relu(x)
        x = self.pool(x) #59*59*32
        
        x = self.conv2(x) #57*57*64
        x = self.relu(x)
        x = self.pool(x) #28*28*64
        
        x = x.view(-1, 28*28*64)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x


# In[14]:


model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)


# In[15]:


def train(model, loss_fn, optimizer, epochs = 15):
    
    training_loss=[]
    training_acc=[]
    validation_loss =[]
    validation_acc=[]
    
    
    for epoch in range(epochs):
        train_loss=0.0
        train_accuracy=0.0
        model.train()
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            
            loss = loss_fn(output, labels)
            
            
            loss.backward()
            optimizer.step()
            prediction = torch.argmax(output,1)
            train_loss +=loss.item()
            train_accuracy+=(prediction==labels).sum().item()
        training_acc.append(train_accuracy/len(train_data))
        training_loss.append(train_loss/len(train_loader))
        
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                output = model(images)
                loss = loss_fn(output, labels)
                prediction = torch.argmax(output,1)
                val_loss+=loss.item()
                val_accuracy+=(prediction==labels).sum().item()
            validation_acc.append(val_accuracy/len(val_data))
            validation_loss.append(val_loss/len(val_loader))
            
        print("Epoch: {} Training Accuracy {:.2f}, Training Loss {:.2f}, Validation Accuracy {:.2f}, Validation Loss {:.2f}".format(
             epoch +1, train_accuracy/len(train_data), train_loss/len(train_loader), val_accuracy/len(val_data), val_loss/len(val_loader)
        ))


# In[16]:


train(model, loss_fn, optimizer, epochs = 20)


# In[22]:


def predict_image(img,model): ## Test Function 
  # convert to a batch of size 1
  xb = img.unsqueeze(0)

  # get prediction
  yb = model(xb) #(batch_size,img_dim)

  pred = torch.argmax(yb,dim=1)

  return img_data.classes[pred]


# In[23]:


img , label = test_data[4]
plt.imshow(img.permute(1,2,0))
print("Actual Label: ",img_data.classes[label] , "Predicted:", predict_image(img,model))


# In[ ]:




