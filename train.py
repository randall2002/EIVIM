import os
import zipfile
import copy
import torch
from IVIM_Dataset import train_dataloader, val_dataloader
import numpy as np
import pandas as pd
from time import time
from torch import nn
import torch.optim as optim
# from criterion import param_loss, img_loss
from functions_and_demo import read_data
from model import U_Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, traindataloader, valdataloader, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss =  float('inf')
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_loss = 0.0
        train_num = 0
        val_loss = 0.0
        val_num = 0
        model.train()     # train modality
        for step, (b_x, b_y) in enumerate(traindataloader):
            optimizer.zero_grad()
            b_x = b_x.float().to(device)
            b_y = b_y.long().to(device)
            out = model(b_x)    # 傅里叶变换后的图像作为输入
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(b_y)
            train_num += len(b_y)
        # 计算一个epoch在训练集上的精度和损失
        train_loss_all.append(train_loss / train_num)
        print('{} Train Loss:{:.4f}'.format(epoch, train_loss_all[-1]))

        # 计算一个epoch训练后在验证集上的损失
        model.eval()
        for step, (b_x, b_y) in enumerate(valdataloader):
            b_x = b_x.float().to(device)
            b_y = b_y.long().to(device)
            out = model(b_x)    # 傅里叶变换后的图像作为输入
            loss = criterion(out, b_y)
            val_loss += loss.item() * len(b_y)
            val_num += len(b_y)
        # 计算一个epoch在验证集上的精度和损失
        val_loss_all.append(val_loss / val_num)
        print('{} Val Loss:{:.4f}'.format(epoch, val_loss_all[-1]))

        # 保存最好的网络参数
        if val_loss_all[-1] < best_loss:
            best_loss = val_loss_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        # 每个epoch花费的时间
        time_use = time.time() - since
        print("Train and val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
    
    train_process = pd.DataFrame(
        data={"epoch":range(num_epochs),
              "train_loss_all":train_loss_all,
              "vall_loss_all":val_loss_all})
    # 输出最好的模型
    model.load_state_dict(best_model_wts)
    return model, train_process


unet = U_Net()
# 定义损失函数和优化器
LR = 0.003
criterion = nn.NLLLoss
optimizer = optim.Adam(unet.parameters(), lr=LR,  weight_decay=0)
# 对模型迭代训练，所有数据训练epoch轮
net, train_process = train_model(U_Net, criterion, optimizer, train_dataloader, val_dataloader, num_epochs=25)
# 保存训练好的网络 U_Net
torch.save(U_Net, "U_Net.pkl")




# if __name__ == '__main__':
#     file_dir='/homes/lwjiang/Data/IVIM/public_training_data/'
#     file_Resultdir='/homes/lwjiang/Data/IVIM/Result'
#     fname_gt ='_IVIMParam.npy'
#     fname_tissue ='_TissueType.npy'
#     fname_noisyDWIk = '_NoisyDWIk.npy'
#     num_cases = 2
#     Nx = 200
#     Ny = 200
#     b = np.array([0, 5, 50, 100, 200, 500, 800, 1000])

#     rRMSE_case =np.empty([num_cases])
#     rRMSE_t_case =np.empty([num_cases])


