import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import os
import zipfile
import numpy as np
from scipy.optimize import curve_fit

# 自定义转换，将 NumPy 数组转换为 PyTorch 张量，并调整通道顺序
class NumpyToTensor(object):
    def __call__(self, sample):
        # 调整通道顺序并转换为张量
        image = np.transpose(sample, (2, 0, 1))
        return torch.from_numpy(image)

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.fname_gt = '_IVIMParam.npy'
        self.fname_tissue = '_TissueType.npy'
        self.fname_noisyDWIk = '_NoisyDWIk.npy'
        self.fname_gtDWIs = '_gtDWIs.npy'
        self.transform = transform

        self.start_index, self.count = self.__get_start_number_and_count(data_dir, self.fname_noisyDWIk)

        print(f"Start Number: {self.start_index}, Total Files: {self.count}")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        noisy_image = self.load_noisy_image(index)
        param_maps = self.load_param_maps(index)
        noiseless_image = self.load_noiseless_image(index)

        # 应用数据增强
        #transform会导致维度顺序发生变化，所以改成一致维度。
        if self.transform:
            angle = transforms.RandomRotation.get_params([-10, 10])  # 随机角度
            flip = torch.rand(1) < 0.5  # 随机决定是否翻转
            #应用变换
            noisy_image = self.apply_transform(noisy_image, angle, flip)
            noiseless_image = self.apply_transform(noiseless_image, angle, flip)
            param_maps = self.apply_transform(param_maps, angle, flip)  # 同样对参数图应用增强
        else:
            numpy_to_tensor = NumpyToTensor()
            noisy_image = numpy_to_tensor(noisy_image)
            noiseless_image = numpy_to_tensor(noiseless_image)
            param_maps = numpy_to_tensor(param_maps)

        # 返回样本和标签数据
        return noisy_image, (param_maps, noiseless_image), index + self.start_index

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

    def apply_transform(self, image, angle, flip):

        # 如果图像是 NumPy 数组，先转换为 PyTorch 张量
        if isinstance(image, np.ndarray):
            numpy_to_tensor = NumpyToTensor()
            image = numpy_to_tensor(image)

        # 应用旋转和翻转变换
        transformed_image = TF.rotate(image, angle)
        if flip:
            transformed_image = TF.hflip(transformed_image)
        return transformed_image



# 测试代码；
import matplotlib.pyplot as plt

# 辅助函数，用于显示图像
def display_sample(noisy_images, noiseless_images, param_maps, sample_index, data_source):
    num_b_values = noisy_images.shape[0]
    fig, axs = plt.subplots(3, max(num_b_values, 3), figsize=(15, 6))

    for i in range(num_b_values):

        axs[0, i].imshow(noisy_images[i, :, :], cmap='gray')
        axs[0, i].set_title(f'Noisy b={i}')
        axs[0, i].axis('off')

        axs[1, i].imshow(noiseless_images[i, :, :], cmap='gray')
        axs[1, i].set_title(f'Noiseless b={i}')
        axs[1, i].axis('off')

    param_names = ['Param1', 'Param2', 'Param3']
    for i in range(3):
        axs[2, i].imshow(param_maps[i, :, :], cmap='gray')
        axs[2, i].set_title(param_names[i])
        axs[2, i].axis('off')

    plt.suptitle(f'Sample Index: {sample_index}, Data Source: {data_source}')
    plt.show()

# 测试代码
def main():
    data_dir = 'E:/Data/public_training_data/training2/'
    transform = transforms.Compose([
        NumpyToTensor(),  # 首先将Numpy数组转化为张量
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        # 可以添加更多的变换...
    ])

    dataset = MyDataset(data_dir, transform=None)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # 加载一个批次的数据并显示一个样本
    for noisy_images, (param_maps, noiseless_images), sample_indices  in dataloader:
        # 只显示第一个样本
        noisy_image = noisy_images[0].numpy()
        noiseless_image = noiseless_images[0].numpy()
        param_map = param_maps[0].numpy()
        sample_index = sample_indices[0]  # 获取第一个样本的索引

        # 显示样本
        display_sample(noisy_image, noiseless_image, param_map, sample_index, data_dir)
        break

if __name__ == '__main__':
    main()

