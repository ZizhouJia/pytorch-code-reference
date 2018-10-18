import torch
import torch.nn as nn
#base resnet 85%
#make x/2 resnet 86.31%
#data_argument and mon 0.9: 93.56%
#after add with a bn data argument and mom 0.9:


class res_block(nn.Module):
    def __init__(self,channels):
        super(res_block,self).__init__()
        self.conv1=nn.Conv2d(channels,channels,3,1,1)
        self.conv2=nn.Conv2d(channels,channels,3,1,1)
        self.bn1=nn.BatchNorm2d(channels)
        self.bn2=nn.BatchNorm2d(channels)
        self.bn3=nn.BatchNorm2d(channels)
        self.relu=nn.ReLU()
    def forward(self,x):
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out=out+x
        out=self.bn3(out)
        out=self.relu(out)
        return out

class res_shortcut(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(res_shortcut,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,3,padding=1,stride=1)
        self.conv2=nn.Conv2d(out_channels,out_channels,3,padding=1,stride=2)
        self.shortcut=nn.Conv2d(in_channels,out_channels,1,padding=0,stride=2)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.bn3=nn.BatchNorm2d(out_channels)
        self.bn_short=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU()

    def forward(self,x):
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out2=self.shortcut(x)
        out2=self.bn_short(out2)
        out=out+out2
        out=self.bn3(out)
        out=self.relu(out)
        return out

class res_body(nn.Module):
    def __init__(self,channels,depth,shortcut=False):
        super(res_body,self).__init__()
        self.layers=[]
        if(shortcut):
            short_cut=res_shortcut(channels/2,channels)
            self.add_module(str(0),short_cut)
            self.layers.append(short_cut)
        else:
            layer=res_block(channels)
            self.add_module(str(0),layer)
            self.layers.append(layer)
        for i in range(0,depth-1):
            layer=res_block(channels)
            self.add_module(str(i+1),layer)
            self.layers.append(layer)

    def forward(self,x):
        out=x
        for i in range(0,len(self.layers)):
            out=self.layers[i](out)
        return out

class res_net(nn.Module):
    def __init__(self,n):
        super(res_net,self).__init__()
        self.conv=nn.Conv2d(3,16,3,1,1)
        self.bn=nn.BatchNorm2d(16)
        self.avg_pool=nn.AvgPool2d(8)
        self.fc=nn.Conv2d(64,10,1,1)
        self.res_body1=res_body(16,n,False)
        self.res_body2=res_body(32,n,True)
        self.res_body3=res_body(64,n,True)

    def forward(self,x):
        out=self.conv(x)
        out=self.bn(out)
        out=self.res_body1(out)
        out=self.res_body2(out)
        out=self.res_body3(out)
        out=self.avg_pool(out)
        out=self.fc(out)
        return out
