# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:40:30 2019

@author: 44266
"""

import torch
import pickle
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from time import time
from prettytable import PrettyTable
# Neural Network Model (2 hidden layer)
class Net(nn.Module):
    def __init__(self, input_s, hidden_s1, hidden_s2, num_class, p1=0.0, 
                 p2=0.0, p3=0.0):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_s, hidden_s1) 
        self.drop_layer1 = nn.Dropout(p=p1)
        self.drop_layer2 = nn.Dropout(p=p2)
        self.drop_layer3 = nn.Dropout(p=p3)
        self.relu = nn.ReLU()        
        self.fc2 = nn.Linear(hidden_s1, hidden_s2)
        self.fc3 = nn.Linear(hidden_s2, num_class)  
    
    def forward(self, x):
        out = self.drop_layer1(x)
        out = self.fc1(out)
        out = self.drop_layer2(out)
        out = self.relu(out)        
        out = self.fc2(out)
        out = self.drop_layer3(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
def run_obj(para=np.random.rand(10), precison=4):
    # set the hyperparameter space    
    hln1 = np.round(2000*para[0]).astype('int')
    hln2 = np.round(2000*para[1]).astype('int')
    lr = np.exp(-10*para[2])
    lr_l = 1 - 0.5*para[3]
    weight_decay = np.exp(-6*para[4])
    dropout1 = para[5]
    dropout2 = para[6]
    dropout3 = para[7]
    momentum = para[8]
    dampening = para[9]
    
    tab_para = PrettyTable()
    tab_para.field_names = ["Name", "Value"]
    tab_para.add_row(['1st hidden layer number', hln1])
    tab_para.add_row(['2nd hidden layer number', hln2])
    tab_para.add_row(['learning rate', np.round(lr, precison)])
    tab_para.add_row(['learning rate decease', np.round(lr_l, precison)])
    tab_para.add_row(['weight_decay', np.round(weight_decay, precison)])
    tab_para.add_row(['dropout prob of input', np.round(dropout1, precison)])
    tab_para.add_row(['dropout prob of 1st layer', np.round(dropout2, precison)])
    tab_para.add_row(['dropout prob of 2st layer', np.round(dropout3, precison)])
    tab_para.add_row(['momentum', np.round(momentum, precison)])
    tab_para.add_row(['dampening', np.round(dampening, precison)])    
    print(tab_para)
    
    val = obj(hln1, hln2, lr, lr_l, weight_decay, dropout1, dropout2,
              dropout3, momentum, dampening)
    return val
def obj(hidden_s1=200, hidden_s2=30, learning_rate=0.01, lr_l=0.95, 
        weight_decay=0, dropout1=0.5, dropout2=0.5, dropout3=0.5, momentum=0.9, 
        dampening=0):
    input_s = 784
    num_class = 10
    num_epochs = 2
    batch_size = 100
    
    tic = time()
    # MNIST Dataset 
    train_dataset = dsets.MNIST(root='../data', train=True, 
                                transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root='../data', train=False, transform=transforms.ToTensor())
    
    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, 
                                              shuffle=False)
    
    
        
    net = Net(input_s, hidden_s1, hidden_s2, num_class, p1=dropout1, 
              p2=dropout2, p3=dropout3)
        
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)  
    optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate, 
                                weight_decay=weight_decay, momentum=momentum, 
                                dampening=dampening)
    # Let learning rate decrease as epoch increase
    lambda1 = lambda epoch: lr_l ** epoch
    scheduler = LambdaLR(optimizer, lr_lambda=[lambda1])
    # Train the Model
    for epoch in range(num_epochs):
        net.train()
        correct = 0
        total = 0
        scheduler.step()
        for i, (images, labels) in enumerate(train_loader):  
            
            # Convert torch tensor to Variable
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(images)        
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                       %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data.item()))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()    
        print('Accuracy of the network on the 60000 train images: %.4f %%' %(100*correct.item()/total))
        # Test the Model
        correct = 0
        total = 0
        net.eval()
        for images, labels in test_loader:
            images = Variable(images.view(-1, 28*28))
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        
        print('Accuracy of the network on the 10000 test images: %.4f %%' %(100*correct.item()/total))
    print("the elapsed time is {}".format(time() - tic))
    return correct.item()/total

def run_cons(para=np.random.rand(10), precison=4):
    # set the hyperparameter space    
    hln1 = np.round(2000*para[0]).astype('int')
    hln2 = np.round(2000*para[1]).astype('int')
    lr = np.exp(-10*para[2])
    lr_l = 1 - 0.5*para[3]
    weight_decay = np.exp(-6*para[4])
    dropout1 = para[5]
    dropout2 = para[6]
    dropout3 = para[7]
    momentum = para[8]
    dampening = para[9]
    
    tab_para = PrettyTable()
    tab_para.field_names = ["Name", "Value"]
    tab_para.add_row(['1st hidden layer number', hln1])
    tab_para.add_row(['2nd hidden layer number', hln2])
    tab_para.add_row(['learning rate', np.round(lr, precison)])
    tab_para.add_row(['learning rate decease', np.round(lr_l, precison)])
    tab_para.add_row(['weight_decay', np.round(weight_decay, precison)])
    tab_para.add_row(['dropout prob of input', np.round(dropout1, precison)])
    tab_para.add_row(['dropout prob of 1st layer', np.round(dropout2, precison)])
    tab_para.add_row(['dropout prob of 2st layer', np.round(dropout3, precison)])
    tab_para.add_row(['momentum', np.round(momentum, precison)])
    tab_para.add_row(['dampening', np.round(dampening, precison)])    
    print(tab_para)
    val = cons(hln1, hln2)
    return val
    
def cons(hidden_s1=20, hidden_s2=20):
    input_s = 784
    num_class = 10
    batch_size = 100
    net = Net(input_s, hidden_s1, hidden_s2, num_class, p1=0.5, p2=0.5, p3=0.5)
    # MNIST Dataset 
    train_dataset = dsets.MNIST(root='../data', train=True, 
                                transform=transforms.ToTensor(), download=True) 
    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                               shuffle=True)
    totaltime = 0
    tic1 = time()
    for _ in range(10):
        time_sum = 0    
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 28*28))
            tic = time()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            toc = time()
            time_sum += toc - tic
            if i == 100:
                break            
        totaltime += time_sum
    print("elapse time is {}".format(time()-tic1))
    return totaltime/10
def get_data():
    sample = np.random.randint(2000, size=(400, 2))
    sample = np.unique(sample, axis=0)    
    cons_out = []
    for i in range(sample.shape[0]):
        cons_out.append(cons(hidden_s1=sample[i, 0], hidden_s2=sample[i, 1]))
        print('{} %'.format(100*i/sample.shape[0]))
    with open('cons.pkl', 'wb') as f:
        pickle.dump([sample, cons_out], f)
def read_data():
    with open('cons.pkl', 'rb') as f:
        [sample, cons] = pickle.load(f)
    print(sample.shape)
    print(len(cons))
def hi(x=2, y=3):
    if x==1:
        print('the value x=1')
    else:
        print(x, y)            
    return "out"

if __name__ == "__main__":
    #get_data()
    #run_obj()
    print(run_cons())
    #obj()
    #run_cons()
    #hi()