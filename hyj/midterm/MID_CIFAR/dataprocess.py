import numpy
import os
import torch
import torch.nn as nn
import torch.functional
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models
import torchvision.transforms as transforms
import torchvision.utils
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import pylab
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
device =  'cuda' if torch.cuda.is_available() else 'cpu'

best_acc =0
start_epoch = 0
print('loading data...')
train_transform = transforms.Compose([
    transforms.RandomCrop(32,padding=4),#随机裁剪
    transforms.RandomHorizontalFlip(),#以p=0.5的概率翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))#归一化
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_dataset = torchvision.datasets.CIFAR10(root='cifar-10-batches-py',train=True,transform=train_transform,download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=128,shuffle=True)
test_dataset = torchvision.datasets.CIFAR10(root='cifar-10-batches-py',train=False,transform=test_transform,download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=100,shuffle=False)
classes = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')
'''
data_iter = iter(train_loader)
images,labels = next(data_iter)
cnt = 0
for image,label in train_loader:
    if cnt>=3:
        break
    print(label)
    img = image[0]
    img = img.numpy()
    img = numpy.transpose(img,(1,2,0))
    cnt+=1
    plt.imshow(img)
    plt.show()

'''
class ImageNet1(nn.Module):
    def __init__(self):
        super(ImageNet1,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.fc1 = nn.Linear(in_features=16*5*5,out_features=120)
        self.fc2 = nn.Linear(in_features=120,out_features=84)
        self.fc3 = nn.Linear(in_features=84,out_features=10)
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = ImageNet1()
net = net.to(device)
print(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)

for epoch in range(start_epoch,start_epoch+200):
    print('\nEpoch: ',epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx,(inputs,targets) in enumerate(train_loader):
        inputs,targets = inputs.to(device),targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _,predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    net.eval()
    test_loss = 0
    correct =0
    total = 0
    with torch.no_grad():
        for batch_idx,(inputs,targets) in enumerate(test_loader):
            inputs,targets = inputs.to(device),targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs,targets)
            test_loss += loss.item()
            _,predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    acc = 100.*correct/total
    if acc > best_acc:
        print('saving...')
        state = {
                'net' : net.state_dict(),
                'acc' :acc,
                'epoch' : epoch
            }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state,'./checkpoint/ckpt.pth')
        best_acc = acc
