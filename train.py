import argparse, os
import zipfile
import copy
import torch
from IVIM_Dataset import MyDataset, NumpyToTensor
import numpy as np
import pandas as pd
import time
from torch import nn
import torch.optim as optim
from criterion import param_loss
from functions_and_demo import read_data
from model import U_Net
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description="PyTorch EIVIM")
parser.add_argument("--traindir", default="/homes/lwjiang/Data/IVIM/public_training_data/training1/", type=str, help="training data path")
parser.add_argument("--validdir", default="/homes/lwjiang/Data/IVIM/public_training_data/training2/", type=str, help="validating data path")

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
        for step, (in_noisy_images, (gt_maps, gt_noiseless_images), _) in enumerate(traindataloader):
            optimizer.zero_grad()
            in_noisy_images = in_noisy_images.float().to(device)
            gt_maps = gt_maps.long().to(device)
            gt_noiseless_images =gt_noiseless_images.float().to(device)


            out = model(in_noisy_images)
            loss = criterion(out, gt_maps)#可能网络需要输出s0,并把S0跟无噪图相比完善loss.
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(gt_maps)
            train_num += len(gt_maps)
        # 计算一个epoch在训练集上的精度和损失
        train_loss_all.append(train_loss / train_num)
        print('{} Train Loss:{:.4f}'.format(epoch, train_loss_all[-1]))

        # 计算一个epoch训练后在验证集上的损失
        model.eval()
        for step, (in_noisy_images, (gt_maps, gt_noiseless_images), _) in enumerate(valdataloader):
            in_noisy_images = in_noisy_images.float().to(device)
            gt_maps = gt_maps.long().to(device)
            gt_noiseless_images = gt_noiseless_images.float().to(device)
            out = model(in_noisy_images)    # 傅里叶变换后的图像作为输入
            loss = criterion(out, gt_maps)
            val_loss += loss.item() * len(gt_maps)
            val_num += len(gt_maps)
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

def main():

    global opt, model
    opt = parser.parse_args()
    print(opt)
    #---------------------------

    train_dir = opt.traindir
    valid_dir = opt.validdir

    transform = transforms.Compose([
        NumpyToTensor(),  # 首先将Numpy数组转化为张量
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        # 可以添加更多的变换...
    ])


    train_dataset = MyDataset(train_dir, transform=None) #数据在线增强暂时不选。
    valid_dataset = MyDataset(valid_dir, transform=None)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

    #--------------------------
    unet = U_Net(in_ch=8, out_ch=3).to(device) #1,设法读取数据后实例化模型；2，需要考虑s0是否送入网络。
    # 定义损失函数和优化器
    LR = 0.003
    criterion = nn.NLLLoss()#这个损失函数要求标签的数据类型为long,...
    optimizer = optim.Adam(unet.parameters(), lr=LR,  weight_decay=0)
    # 对模型迭代训练，所有数据训练epoch轮
    net, train_process = train_model(unet, criterion, optimizer, train_dataloader, valid_dataloader, num_epochs=25)
    # 保存训练好的网络 U_Net
    torch.save(net.state_dict(), "U_Net.pkl")




if __name__ == '__main__':
    main()
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


