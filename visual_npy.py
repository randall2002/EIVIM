import numpy as np
import matplotlib.pyplot as plt
import os

depthmap = np.load('/homes/lwjiang/Data/IVIM/pred_val_bymyself/0001_IVIMParam.npy')    #使用numpy载入npy文件
print("深度图的形状:", depthmap.shape)  

target_dir = '/homes/lwjiang/Data/IVIM/pred_val_bymyself'
for i in range(3):
    plt.imshow(depthmap[:, :, i], cmap='grey')              #执行这一行后并不会立即看到图像，这一行更像是将depthmap载入到plt里
    plt.savefig(os.path.join(target_dir, f'depthmap{i+1}.jpg'))       #执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
    plt.show()                        #在线显示图像

