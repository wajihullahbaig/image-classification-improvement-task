#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:40:56 2022

@author: wajihullah.baig
"""

import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)

random_seed = 1 # or any of your favorite number
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True


from custom_dataset import TrainDataSet, GistTrainDataSet

batch_size = 32
test_split = .3
shuffle_dataset = True
random_seed = 42
n_epochs = 100

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in loader:            
            x = x.to(device=device)           
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    # Toggle model back to train
    model.train()
    return 100.*num_correct / num_samples

def run_gist_features_train_test():

    
    dataset = GistTrainDataSet("dataset/train_gist_augumented2_labels.csv",
                            "dataset")
    
    labels = []
    input_length = 0
    try:
        for feature, label in dataset:
            labels.append(label)
            input_length = len(feature)
    except:
        pass        
    # Since we dont have a test set  with labels to validate the model
    # we split the data into training and testing set
    
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
    # Stratified split    
    train_indices,test_indices,_,_ = train_test_split(indices,labels, test_size=test_split,random_state=random_seed ) 
    
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=test_sampler)
    
    from nn_model import MLP
    
    
    
    model = MLP(input_length).to(device)
    
    # specify loss function
    criterion = nn.CrossEntropyLoss()
    
    # specify optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    

    
    model.train() # prep model for training
    
    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for data, target in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.to(device))
            # calculate the loss
            loss = criterion(output, target.to(device))
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*data.size(0)
            
        # print training statistics 
        # calculate average loss over an epoch
        train_loss = train_loss/len(train_loader.dataset)
    
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch+1, 
            train_loss
            ))
        
        
    print("GIST MLP Train accuracy:",str(check_accuracy(train_loader, model).item())  +"%")     
     # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(4))
    class_total = list(0. for i in range(4))
    
    model.eval() # prep model for *evaluation*
    
    for data, target in test_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data.to(device))
        # calculate the loss
        loss = criterion(output, target.to(device))
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.to(device).data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(min(batch_size,data.size()[0])):   
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    
    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    
    for i in range(4):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (class_total[i]))
    
    print('\nGIST MLP Test Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))   
                                                                      
    torch.save(model,"models/gist_1d_320_vector_mlp.net")
                                
def run_raw_features_train_test():
    


    transform = transforms.Compose([transforms.ToTensor(), 
                                    
        transforms.Lambda(lambda x: torch.flatten(x))]
                                   )
    
    dataset = TrainDataSet("dataset/train_augumented2_labels.csv",
                            "dataset", transform)
    
    labels = []
    input_length = 0
    try:
        for feature, label in dataset:
            labels.append(label)
            input_length = len(feature)
    except:
        pass        
    # Since we dont have a test set  with labels to validate the model
    # we split the data into training and testing set
    

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
    # Stratified split    
    train_indices,test_indices,_,_ = train_test_split(indices,labels, test_size=test_split,random_state=random_seed ) 
    
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=test_sampler)
    
    from nn_model import MLP
    model = MLP(input_length).to(device)
    
    # specify loss function
    criterion = nn.CrossEntropyLoss()
    
    # specify optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    
 
    
    model.train() # prep model for training
    
    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for data, target in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.to(device))
            # calculate the loss
            loss = criterion(output, target.to(device))
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*data.size(0)
            
        # print training statistics 
        # calculate average loss over an epoch
        train_loss = train_loss/len(train_loader.dataset)
    
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch+1, 
            train_loss
            ))
        
    print("Raw MLP Train accuracy:",str(check_accuracy(train_loader, model).item()) +"%" )     
        
     # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(4))
    class_total = list(0. for i in range(4))
    
    model.eval() # prep model for *evaluation*
    
    for data, target in test_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data.to(device))
        # calculate the loss
        loss = criterion(output, target.to(device))
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.to(device).data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(min(batch_size,data.size()[0])):   
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    
    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    
    for i in range(4):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (class_total[i]))
    
    print('\nRAW MLP Test Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))   
    
    
    torch.save(model,"models/raw_1d_784_vector_mlp.net")
if __name__ == '__main__':
    run_raw_features_train_test()
    run_gist_features_train_test()