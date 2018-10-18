import torch
import torch.nn as nn
import optimizer

def train_cifar(cnn,epochs,train_loader,test_loader,level=[50,75,100],rate=[0.1,0.01,0.001],test=True):
    optimizers=optimizer.get_SGD_optimizers(list(cnn.parameters()),rate)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(0,epochs):
        loss_value=100.0
        cnn.train()
        for step, (x, y) in enumerate(train_loader):
            Xu=x.cuda()
            Yu=y.cuda()
            output = cnn(Xu)
            output=output.view(output.size(0), -1)
            loss = loss_function(output,Yu)
            loss_value=loss.cpu().item()
            loss.backward()
            optimizer.optimize(optimizers,level,epoch)
            if(step%100==0):
                print("in epoch %d step %d"%(epoch,step))
        print("the loss in epoch %d is %.4f"%(epoch,loss))
        if(test):
            correct=0.0
            for data, target in test_loader:
                cnn.eval()
                data, target = data.cuda(), target.cuda()
                output = cnn(data)
                output=output.view(output.size(0), -1)
                pred = torch.max(output,1)[1]
                curr=torch.sum((pred==target).float())
                correct +=curr.cpu().data.item()
            accuracy=correct/len(test_loader.dataset)
            print("the test acc in epoch %d is %.4f"%(epoch,accuracy))
