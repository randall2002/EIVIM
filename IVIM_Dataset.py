import torch
from torch.utils.data import Dataset, DataLoader
import os
import zipfile
import numpy as np
from scipy.optimize import curve_fit

class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.fname_gt = '_IVIMParam.npy'
        self.fname_tissue = '_TissueType.npy'
        self.fname_noisyDWIk = '_NoisyDWIk.npy'
        self.fname_gtDWIs = '_gtDWIs.npy'

        self.start_index, self.count = self.__get_start_number_and_count(data_dir, self.fname_noisyDWIk)

        print(f"Start Number: {self.start_index}, Total Files: {self.count}")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        noisy_image = self.load_noisy_image(index)
        param_maps = self.load_param_maps(index)
        noiseless_image = self.load_noiseless_image(index)

        # 返回样本和标签数据
        return noisy_image, (param_maps, noiseless_image)

    def load_noisy_image(self, index):
        # 加载带噪声的图像
        i = index + self.start_index

        noisy_k = self.__read_data(self.data_dir, self.fname_noisyDWIk, i)
        # Assuming you have a function to perform Fourier transform on x
        noisy_image = np.abs(np.fft.ifft2(noisy_k, axes=(0, 1), norm='ortho'))
        noisy_preprocessed = self.__preprocess_1bseries(noisy_image)
        return noisy_preprocessed

    def __preprocess_1bseries(self, noisy_image):

        pass
        return noisy_image

    def load_param_maps(self, index):
        # 加载参数图
        i = index + self.start_index
        data = self.__read_data(self.data_dir, self.fname_gt, i)
        param_maps = data
        return param_maps

    def load_noiseless_image(self, index):
        # 加载无噪声空间图
        i = index + self.start_index
        noiseless_image = np.abs(self.__read_data(self.data_dir, self.fname_gtDWIs, i))
        return noiseless_image

    def __get_start_number_and_count(self, data_dir, file_name):
        # 获取所有符合条件的文件名
        files = [f for f in os.listdir(data_dir) if file_name in f]

        # 提取文件序号并转换为整数
        numbers = [int(f.split('_')[0]) for f in files]

        # 如果没有找到文件，返回None
        if not numbers:
            return None, None

        # 获取起始号和文件总数
        start_number = min(numbers)
        count = len(numbers)

        return start_number, count

    def __read_data(self, file_dir, fname, i):
        fname_tmp = file_dir + "{:04}".format(i) + fname
        data = np.load(fname_tmp)
        return data



# 测试代码；
import matplotlib.pyplot as plt

<<<<<<< HEAD
# 辅助函数，用于显示图像
def display_sample(noisy_images, noiseless_images, param_maps):
    num_b_values = noisy_images.shape[2]
    fig, axs = plt.subplots(3, max(num_b_values, 3), figsize=(15, 6))

    for i in range(num_b_values):
        axs[0, i].imshow(noisy_images[:, :, i], cmap='gray')
        axs[0, i].set_title(f'Noisy b={i}')
        axs[0, i].axis('off')

        axs[1, i].imshow(noiseless_images[:, :, i], cmap='gray')
        axs[1, i].set_title(f'Noiseless b={i}')
        axs[1, i].axis('off')

    param_names = ['Param1', 'Param2', 'Param3']
    for i in range(3):
        axs[2, i].imshow(param_maps[:, :, i], cmap='gray')
        axs[2, i].set_title(param_names[i])
        axs[2, i].axis('off')

    plt.show()

# 测试代码
def main():
    data_dir = 'E:/Data/public_training_data/training2/'
    dataset = MyDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 加载一个批次的数据并显示一个样本
    for noisy_images, (param_maps, noiseless_images) in dataloader:
        # 只显示第一个样本
        noisy_image = noisy_images[0].numpy()
        noiseless_image = noiseless_images[0].numpy()
        param_map = param_maps[0].numpy()

        # 显示样本
        display_sample(noisy_image, noiseless_image, param_map)
        break

if __name__ == '__main__':
    main()

=======
# Create validation dataset and dataloader
val_dataset = MyDataset(file_dir, fname_noisyDWIk, num_train_cases + 1, num_train_cases + num_val_cases)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
>>>>>>> origin/main
