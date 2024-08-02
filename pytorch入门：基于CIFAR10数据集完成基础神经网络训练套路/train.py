import torchvision
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from network import Net


#准备数据集
dataset_train=torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)
dataset_test=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)


#加载数据集
loader_train=DataLoader(dataset_train,batch_size=16,drop_last=True,shuffle=False)
loader_test=DataLoader(dataset_test,batch_size=16,drop_last=True,shuffle=False)

#搭建神经网络
net=Net()


#创建损失函数
loss_cro=torch.nn.CrossEntropyLoss()

#创建优化器
optim=torch.optim.SGD(net.parameters(),lr=1e-3)

#设置训练网络的一些参数
total_train_stp=0#总训练次数
total_test_stp=0#总测试次数
epoch=20#训练轮次

#tensorboard
writer=SummaryWriter("./end")

#开始训练
for i in range(epoch):
    loss=0
    print("-----------第{}轮训练开始--------------".format(i+1))
    for data in loader_train:
        imgs,targets=data
        output=net(imgs)
        loss=loss_cro(output,targets)

        #优化器优化
        optim.zero_grad()
        loss.backward()
        optim.step()

        #记录训练次数
        total_train_stp+=1
        if total_train_stp%100==0:
            print("训练次数：{}，loss:{}".format(total_train_stp,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_stp)

#测试
with torch.no_grad():
    total_current_test=0
    for data in loader_test:
        imgs,targets=data
        output=net(imgs)
        loss=loss_cro(output,targets)
        total_test_stp+=1

        writer.add_scalar("test_loss",loss,total_test_stp)
        accuracy=(output.argmax(1)==targets).sum()
        total_current_test+=accuracy
        print("整体测试的正确率为{}".format(total_current_test/(total_test_stp)*10))
        writer.add_scalar("test_accuracy",total_current_test/(total_test_stp)*10,total_test_stp)
torch.save(net,"./model3.pth")
print("模型已经保存")

writer.close()

