# Author: Little-Chen
# Emial: Chenxiuyan_t@163.com
import torch.nn as nn
import logging
import torch.nn.functional as F
from utils.evaluate import*
from matplotlib import pyplot as plt
import os

def train(model, optimizer, train_dataloader, test_dataloader, device, args):
     # set loss function
    classification_loss = nn.CrossEntropyLoss()  
    model.train()

    total_loss = []
    total_test_acc = []

    for epoch in range(args.epoch):

        for batch_idx, (data, target) in enumerate(train_dataloader):
            
            model.train()

            data, target = data.to(device), target.to(device)
            model.zero_grad()
            
            y_hat = model(data)
            y_hat = F.log_softmax(y_hat)
            # print("y_hat_shape", y_hat.shape)
            loss = classification_loss(y_hat, target)
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())

            
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataloader)*args.batch_size_train,
                100. * batch_idx / len(train_dataloader), total_loss[-1]))
                
            test_acc = evaluate(model, device, test_dataloader)

            total_test_acc.append(test_acc)
        
        logging.info('\n')

    # 画图
    plt.plot(total_loss)
    plt.title("Loss")
    plt.savefig(args.results_path+'/loss.png')
    plt.cla()

    plt.plot(total_test_acc)
    plt.title("Test acc")
    plt.savefig(args.results_path+'/Test_acc.png')
    plt.cla()

    # 保存模型
    torch.save(model, args.results_path+'/CNN.pth')