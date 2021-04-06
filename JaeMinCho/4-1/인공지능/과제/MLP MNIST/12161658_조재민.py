#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch
import torchvision
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms


# In[14]:


num_workers = 0 # 데이터 불러오는데 사용 될 하위 프로세서 수 
batch_size_train = 64 # 각 batch마다 몇 개의 샘플을 받아올지..
batch_size_val = 64
batch_size_test = 1000
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='mnist', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='mnist', train=False, download=True, transform=transform)

train_data, val_data = torch.utils.data.random_split(train_data,[50000,10000])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size_val, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, num_workers=num_workers)

dataiter = iter(train_loader)
images, labels = dataiter.next()


# In[15]:


import torch.nn as nn
import torch.nn.functional as F

#NN 아키텍쳐 정의 !
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(28 * 28, 64) 
        self.fc2 = nn.Linear(64, 64) # n_hidden -> hidden_2
        self.fc3 = nn.Linear(64, 10) # n_hidden -> 10
        self.dropout = nn.Dropout(0.2) # dropOut Layer p = 0.2, overfitting 을 피하기 위해

    def forward(self, x):
        x = x.view(-1, 28 * 28) # 이미지 평평하게 만들기
        x = F.relu(self.fc1(x)) # relu 함수 사용해서 hidden layer 추가 ..
        return x

model = Net() # NN 초기화


# In[16]:


criterion = nn.CrossEntropyLoss() # Loss Function 정의
images, labels = next(iter(train_loader))
images = images.view(images.shape[0],-1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9) # Optimizer 정의


# In[19]:


n_epochs = 150 # Train할 epoch 개수

model.train() # Train 준비

for epoch in range(n_epochs):

    running_loss = 0 # running loss 초기화
    
    # 여기서 Train !
    for data, target in train_loader:
        optimizer.zero_grad()# Optimized 변수들의 gradients 초기화
        
        output = model(data) # Forward Pass! data를 model에 넣어서 예측값 계산
        loss = citerion(output, target) # loss 계산 

        loss.backward() # model의 값과 loss로 gradient 계산
        optimizer.step() # parameter를 업데이트한다. optimization step
        
        running_loss += loss.item()/len(train_loader)
    else:
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.view(images.shape[0], -1)
                log_ps = model(images)
                val_loss += criterion(log_ps, labels)
            
    print("Epoch {} - Trainning Loss: {} Val Loss: {}".format(epoch, running_loss, val_loss/len(val_loader)))


# In[21]:


#Test Loss 와 accuracy (정확도) 계산

test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # model evaluate 할 준비

for data, target in test_loader:
    output = model(data) # 예측된 결과물을 model에 input해 계산한다

    loss = criterion(output, target) # loss 계산 
    test_loss += loss.item()*data.size(0) # test loss 최신화 
    _, pred = torch.max(output, 1) # 예측된 클래스로 output 변환 
    
    correct = np.squeeze(pred.eq(target.data.view_as(pred))) # 예측값과 비교 
    
    for i in range(batch_size_val): # 각 object의 정확도를 계산한다 
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

test_loss = test_loss/len(test_loader.dataset) 
print('Test Loss: {:.6f}\n'.format(test_loss)) # 출력

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nModel Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))


# In[ ]:




