import argparse, os
import zipfile
import copy
import time
import torch
import random
from IVIM_Dataset import MyDataset, NumpyToTensor
import numpy as np
import pandas as pd
import time
from torch import nn
import torch.optim as optim
from criterion import param_loss, rRMSE_per_batch
from functions_and_demo import read_data
from model_unet import U_Net
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="PyTorch EIVIM")
parser.add_argument("--traindir", default="/homes/lwjiang/Data/IVIM/public_training_data/training1/", type=str, help="training data path")
parser.add_argument("--validdir", default="/homes/lwjiang/Data/IVIM/public_training_data/training2/", type=str, help="validating data path")
parser.add_argument("--batchsize", default=4, type=int)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom loss function
def custom_loss(output, gt_maps, gt_noiseless_images, alpha):
    gt_s0_images = gt_noiseless_images[:, 0, :, :]
    criterion = nn.MSELoss(reduction='mean')
    param_loss = criterion(output[:, :3, :, :], gt_maps)
    s0_loss = criterion(output[:, 3, :, :], gt_s0_images)
    return (1 - alpha) * param_loss + alpha * s0_loss

def train_model(model, optimizer, traindataloader, valdataloader, num_epochs=25):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss =  float('inf')
    best_out = []
    train_loss_all = []
    train_acc_all = []
    train_rRMSE_all = []
    val_loss_all = []
    val_acc_all = []
    val_rRMSE_all = []
    output_dir = f"/homes/lwjiang/Data/IVIM/public_training_data/result_{time.strftime('%M%s')}"
    writer = SummaryWriter(output_dir)
    #alpha为损失函数的调节参数，暂时固定。
    alpha = 0
    for epoch in range(num_epochs):
        since = time.time()
        print('-' * 40)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        train_loss_case, train_rRMSE_case, out_param = do_train_for_every_epoch(model, alpha, optimizer, traindataloader)
        # 把每一次epoch在训练集上的样本平均损失 添加到列表里
        train_loss_all.append( train_loss_case )
        train_rRMSE_all.append(train_rRMSE_case)
        print('{} Train Loss:{:.4f}'.format(epoch, train_loss_case))

        time_use1 = time.time() - since
        print("train complete in {:.0f}m {:.0f}s".format(time_use1 // 60, time_use1 % 60))
        since2 = time.time()

        val_loss_case, val_rRMSE_case = do_evla_for_every_epoch(model, alpha, valdataloader)
        writer.add_scalar('Loss/train', train_loss_case, epoch)
        writer.add_scalar('Loss/valid', val_loss_case, epoch)
        writer.add_scalar('Metric/train', train_rRMSE_case, epoch)
        writer.add_scalar('Metric/valid', val_rRMSE_case, epoch)
         
        # 计算一个epoch在验证集上的精度和损失
        val_loss_all.append(val_loss_case)
        val_rRMSE_all.append(val_rRMSE_case)
        print('{} Val Loss:{:.4f}'.format(epoch, val_loss_case))

        # 保存最好的网络参数
        if val_loss_all[-1] < best_loss:
            best_loss = val_loss_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        
            # 保存最好的网络输出参数图
            best_out = out_param
        
        # 每个epoch花费的时间
        time_use2 = time.time() - since2
        print("val complete in {:.0f}m {:.0f}s".format(time_use2 // 60, time_use2 % 60))

        time_use = time.time() - since
        print("Train and val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
    
    train_process = pd.DataFrame(
        data={"epoch":range(num_epochs),
              "train_loss_all":train_loss_all,
              "val_loss_all":val_loss_all,
              "train_rRMSE_all": train_rRMSE_all,
              "val_rRMSE_all": val_rRMSE_all})

    # 使用detach方法将张量从计算图中分离出来，然后将其移动到CPU上，并转换为NumPy数组          
    best_out_cpu = best_out.detach().cpu().numpy()

    # 输出最好的模型和预测参数图
    model.load_state_dict(best_model_wts)
    np.save('/homes/lwjiang/Data/IVIM/public_training_data/result/param.npy', best_out_cpu)
    writer.close()
    return model, train_process


def do_train_for_every_epoch(model, alpha, optimizer, traindataloader):
    train_loss = 0.0#训练集上的总损失；
    train_rRMSE = 0.0#训练集上的总rRMSE
    train_count = 0#训练集总样本数, num会有歧义（能表示序号和总数）
    model.train()  # train modality
    for step, batch_data in enumerate(tqdm(traindataloader)):
        in_noisy_images, (gt_maps, gt_noiseless_images, tissue_image), _ = batch_data
        #print(gt_noiseless_images.size())
        optimizer.zero_grad()
        in_noisy_images = in_noisy_images.to(device)
        gt_maps = gt_maps.to(device)
        gt_noiseless_images = gt_noiseless_images.to(device)

        out = model(in_noisy_images)
        out_param = out
        #print(out.size())
        loss = custom_loss(out, gt_maps, gt_noiseless_images, alpha)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(gt_maps)
        train_count += len(gt_maps)
        #--------------------------------------
        rRMSE, rRMSE_t = rRMSE_per_batch(out.detach(), gt_maps.detach(), tissue_image.detach())
        train_rRMSE += rRMSE.item() * len(gt_maps)
        #--------------------------

    return train_loss/train_count, train_rRMSE/train_count, out_param #返回平均损失, 平均rRMSE, 三幅预测的参数图

def do_evla_for_every_epoch(model, alpha, valdataloader):
    val_loss = 0.0#验证集上的总损失
    val_rRMSE = 0.0#验证集上的总rRMSE
    val_count = 0 #验证集样本数
    model.eval()
    with torch.no_grad():
        for step, batch_data in enumerate(tqdm(valdataloader)):
            in_noisy_images, (gt_maps, gt_noiseless_images, tissue_image), _ = batch_data
            in_noisy_images = in_noisy_images.to(device)
            gt_maps = gt_maps.to(device)
            gt_noiseless_images = gt_noiseless_images.to(device)
            out = model(in_noisy_images)  # 傅里叶变换后的图像作为输入
            loss = custom_loss(out, gt_maps, gt_noiseless_images, alpha)
            val_loss += loss.item() * len(gt_maps)
            val_count += len(gt_maps)

            # --------------------------------------
            #对比评估标准。
            rRMSE, rRMSE_t = rRMSE_per_batch(out.detach(), gt_maps.detach(), tissue_image.detach())
            val_rRMSE += rRMSE.item() * len(gt_maps)
            # --------------------------

    print("val count:", val_count, "gt map len:", len(gt_maps))
    return val_loss/val_count, val_rRMSE/val_count

def save_net_train_process(net, train_process, train_dir):
    #规范路径：
    norm_train_dir1 = os.path.normpath(train_dir)
    # 构建结果目录路径
    net_dir = os.path.join(os.path.dirname(norm_train_dir1), "net")
    # 确保结果目录存在，如果不存在，则创建它
    if not os.path.exists(net_dir):
        os.makedirs(net_dir)
    net_path = os.path.join(net_dir, "U_net.pkl")
    torch.save(net.state_dict(), net_path)

    #-----------------------------------------------------
    # 构建结果目录路径
    result_dir = os.path.join(os.path.dirname(norm_train_dir1), "result")
    # 确保结果目录存在，如果不存在，则创建它
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 构建最终的CSV文件路径
    train_process_result = os.path.join(result_dir, "result.csv")

    # 将DataFrame保存到CSV文件
    train_process.to_csv(train_process_result, index=False)
##################################################

def main():

    global opt, model
    opt = parser.parse_args()
    print(opt)

    #--------------------------
    #
    seed = 143 #设置固定种子，观察程序重复性
    #seed = random.randint(1, 10000)#使用随机种子，程序每次重新打乱数据
    torch.manual_seed(seed)
    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        torch.cuda.manual_seed(seed)
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

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batchsize, shuffle=False)

    #--------------------------
    unet = U_Net(in_ch=8, out_ch=4).to(device) #1,设法读取数据后实例化模型；2，需要考虑s0是否送入网络。
    #定义损失函数和优化器
    LR = 0.003

    optimizer = optim.Adam(unet.parameters(), lr=LR,  weight_decay=0)
    # 对模型迭代训练，所有数据训练epoch轮
    net, train_process = train_model(unet, optimizer, train_dataloader, valid_dataloader, num_epochs=300)
    save_net_train_process(net, train_process, train_dir)

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