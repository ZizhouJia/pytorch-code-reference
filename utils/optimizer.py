import torch
def get_SGD_optimizers(param,rate):
    optimizers=[]
    for i in range(0,len(rate)):
        optimizer = torch.optim.SGD(param, lr=rate[i],weight_decay=0.0001,momentum=0.9)
        optimizers.append(optimizer)
    return optimizers

def optimize(optimizers,level_thresh,index):
    for i in range(0,len(optimizers)):
        if(index<=level_thresh[i]):
            optimizers[i].step()
            optimizers[i].zero_grad()
            break
